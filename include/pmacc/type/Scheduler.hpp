/* Copyright 2019 Michael Sippel
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <rmngr/scheduler/scheduler_singleton.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/graphviz.hpp>
#include <rmngr/scheduler/fifo.hpp>

namespace pmacc
{

std::ostream& functor_backtrace(std::ostream& out);

template <
    typename Job
>
struct PMaccDispatch : rmngr::DefaultJobSelector<Job>
{
    struct Selector : rmngr::FIFO<Job>
    {
        PMaccDispatch * parent;

        void update()
        {
            parent->update();
        }
    };

    Selector main_selector;
    Selector mpi_selector;

    PMaccDispatch()
    {
        main_selector.parent = this;
        mpi_selector.parent = this;
    }

    struct Property : rmngr::FIFO<Job>::Property
    {
        Property()
	    : dont_schedule_me(false)
	    , mpi_thread(false)
        {}

        bool dont_schedule_me;
        bool mpi_thread;
    };

    void push( Job const & j, Property const & prop = Property() )
    {
        if(! prop.dont_schedule_me )
        {
            if(prop.mpi_thread)
                mpi_selector.push(j, prop);
            else
                main_selector.push(j, prop);
        }
    }

    void notify()
    {
        main_selector.notify();
        mpi_selector.notify();
    }

    bool empty()
    {
        return main_selector.empty() && mpi_selector.empty();
    }

    template <typename Pred>
    Job getJob( Pred const & pred )
    {
        if( rmngr::thread::id == 0 )
            return mpi_selector.getJob( pred );
        else
            return main_selector.getJob( pred );
    }
};

using GraphvizPolicy = rmngr::GraphvizWriter< rmngr::DispatchPolicy< PMaccDispatch >::RuntimeProperty >;

template <typename T>
struct EnqueuePolicy
{
    static bool is_serial(T const & a, T const & b)
    {
        return rmngr::ResourceUser::is_serial(
                   a->template proto_property< rmngr::ResourceUserPolicy >(),
		   b->template proto_property< rmngr::ResourceUserPolicy >());
    }
    static void assert_superset(T const & super, T const & sub)
    {
        auto r_super = super->template proto_property< rmngr::ResourceUserPolicy >();
        auto r_sub = sub->template proto_property< rmngr::ResourceUserPolicy >();
        if(! rmngr::ResourceUser::is_superset( r_super, r_sub ))
        {
            std::stringstream stream;
            stream << "Not allowed: " << std::endl
	           << super->template proto_property< GraphvizPolicy >().label
		   << r_super << std::endl
		   << "is no superset of "
		   << sub->template proto_property< GraphvizPolicy >().label << std::endl
	           << r_sub << std::endl;
            functor_backtrace(stream);
            throw std::runtime_error(stream.str());
        }
    }
};


template <typename Graph>
using RefinementGraph = rmngr::QueuedPrecedenceGraph< Graph, EnqueuePolicy >;

using Scheduler = rmngr::SchedulerSingleton<
    boost::mpl::vector<
        rmngr::ResourceUserPolicy,
        GraphvizPolicy,

        // dispatcher should always be the last policy
        rmngr::DispatchPolicy< PMaccDispatch >
    >,
    RefinementGraph
>;

std::ostream& functor_backtrace(std::ostream& out)
{
    if( std::experimental::optional<std::vector<Scheduler::Schedulable*>> bt = Scheduler::getInstance().backtrace() )
    {
        int i = 0;
        for( auto s : *bt )
        {
            out << "functor backtrace [" << i << "] " << s->proto_property<GraphvizPolicy>().label << std::endl;
            i++;
        }
    }
    return out;
}    

} // namespace pmacc

