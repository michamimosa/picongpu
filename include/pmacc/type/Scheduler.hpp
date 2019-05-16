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
#include <rmngr/scheduler/fifo.hpp>

namespace pmacc
{

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
        {
            std::cout << "run mpi task.." << std::endl;
            return mpi_selector.getJob( pred );
	}
        else
            return main_selector.getJob( pred );
    }
};

using GraphvizPolicy = rmngr::GraphvizWriter< rmngr::DispatchPolicy< PMaccDispatch >::RuntimeProperty >;

template <typename Graph>
using RefinementGraph = rmngr::QueuedPrecedenceGraph< Graph, rmngr::ResourceEnqueuePolicy >;

using Scheduler = rmngr::SchedulerSingleton<
    boost::mpl::vector<
        rmngr::ResourceUserPolicy,
        GraphvizPolicy,

        // dispatcher should always be the last policy
        rmngr::DispatchPolicy< PMaccDispatch >
    >,
    RefinementGraph
>;

} // namespace pmacc

