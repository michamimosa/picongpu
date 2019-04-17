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
  struct PMaccDispatch : rmngr::FIFO<Job>
  {

    struct Property : rmngr::FIFO<Job>::Property
    {
        Property() : dont_schedule_me(false) {}
        bool dont_schedule_me;
    };

    void push( Job const & j, Property const & prop = Property() )
    {
        if(! prop.dont_schedule_me )
            this->rmngr::FIFO<Job>::push( j, prop );
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
    static bool is_superset(T const & super, T const & sub) {
        return rmngr::ResourceUser::is_superset(
                   super->template proto_property< rmngr::ResourceUserPolicy >(),
		   sub->template proto_property< rmngr::ResourceUserPolicy >());
    }
};

template <typename Graph>
using RefinementGraph = rmngr::QueuedPrecedenceGraph< Graph, EnqueuePolicy< typename Graph::vertex_property_type > >;

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

