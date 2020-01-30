/* Copyright 2019-2020 Michael Sippel
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

#include <redGrapes/scheduler/states.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/property/inherit.hpp>
#include <redGrapes/property/resource.hpp>
#include <redGrapes/property/label.hpp>
#include <redGrapes/graph/recursive_graph.hpp>
#include <redGrapes/graph/scheduling_graph.hpp>
#include <redGrapes/thread/thread_dispatcher.hpp>

#include <redGrapes/helpers/cuda/stream.hpp>
#include <redGrapes/helpers/cuda/synchronize_event.hpp>
#include <redGrapes/helpers/mpi/request_pool.hpp>
#include <redGrapes/manager.hpp>

namespace pmacc
{

struct PMaccProperties
{
    bool mpi_task;

    PMaccProperties()
        : mpi_task( false )
    {}

    template < typename PropertiesBuilder >
    struct Builder
    {

        PropertiesBuilder & builder;

        Builder( PropertiesBuilder & b )
            : builder(b)
        {}

        PropertiesBuilder mpi_task()
        {
            builder.prop.mpi_task = true;
            return builder;
        }
    };

    struct Patch
    {
        template <typename PatchBuilder>
        struct Builder
        {
            Builder( PatchBuilder & ) {}
        };
    };

    void apply_patch( Patch const & ) {};
};

using TaskProperties = redGrapes::TaskProperties<
    redGrapes::ResourceProperty,
    redGrapes::LabelProperty,
    PMaccProperties
>;

std::ostream& functor_backtrace(std::ostream& out);

struct EnqueuePolicy
{
    static bool is_serial(TaskProperties const & a, TaskProperties const & b)
    {
        return redGrapes::ResourceUser::is_serial( a, b );
    }

    static void assert_superset(TaskProperties const & super, TaskProperties const & sub)
    {
        if(! redGrapes::ResourceUser::is_superset( super, sub ))
        {
            std::stringstream stream;
            stream << "Not allowed: " << super.label << "is no superset of " << sub.label << std::endl
                   << super.label << " has access: " << std::endl
                   << (redGrapes::ResourceUser)super << std::endl << std::endl
                   << sub.label << " has access: " << std::endl
	           << (redGrapes::ResourceUser)sub << std::endl;
            functor_backtrace(stream);
            throw std::runtime_error(stream.str());
        }
    }
};

template < typename TaskID, typename TaskPtr, typename PrecedenceGraph >
struct Scheduler
    : redGrapes::StateScheduler< TaskID, TaskPtr, PrecedenceGraph >
{
    using typename redGrapes::StateScheduler<TaskID, TaskPtr, PrecedenceGraph>::Job;

    std::mutex queue_mutex;
    std::queue< Job > main_queue;
    std::queue< Job > mpi_queue;

    bool write_graph;

    Scheduler( std::shared_ptr<PrecedenceGraph> pg, size_t n_threads )
        : redGrapes::StateScheduler< TaskID, TaskPtr, PrecedenceGraph >( pg, n_threads )
        , write_graph( false )
    {
        for( size_t i = 0; i < this->schedule.size(); i++ )
        {
            auto & t = this->schedule[i];
            if( i == 1 )
                t.set_request_hook( [this,&t]{ get_job(t, mpi_queue); });
            else
                t.set_request_hook( [this,&t]{ get_job(t, main_queue); });
        }
    }

private:
    void update_queues()
    {
        if( !this->uptodate.test_and_set() )
        {
            auto ready_tasks = this->update_graph();

            for( Job job : ready_tasks )
            {
                auto & prop = job.task_ptr.locked_get();

                if( prop.mpi_task )
                    mpi_queue.push( job );
                else
                    main_queue.push( job );
            }
        }

        if( write_graph )
        {
            static int step=0;
            std::cout << "write step_" << step << ".dot" << std::endl;
            std::ofstream out( "step_" + std::to_string(step++) + ".dot" );
            this->precedence_graph->write_dot(
                out,
                [this]( auto const & task )
                {
                    return task.task_id;
                },
                [this]( auto const & task )
                {
                    return "[" + std::to_string(task.task_id) + "] " + task.label;
                },
                [this]( auto const & task )
                {
                    switch( this->get_task_state(task.task_id) )
                    {
                    case redGrapes::TaskState::uninitialized:
                        return "purple";
                    case redGrapes::TaskState::pending:
                        return "brown";
                    case redGrapes::TaskState::ready:
                        return "green";
                    case redGrapes::TaskState::running:
                        return "yellow";
                    case redGrapes::TaskState::done:
                        return "gray";
                    }
                    return "blue";
                });
        }
    }

    void get_job( redGrapes::ThreadSchedule<Job> & thread, std::queue<Job> & queue )
    {
        std::lock_guard< std::mutex > lock( queue_mutex );

        if( queue.empty() )
            update_queues();

        if( ! queue.empty() )
        {
            auto job = queue.front();
            queue.pop();

            thread.push( job );
        }
    }

};
    
} // namespace pmacc

