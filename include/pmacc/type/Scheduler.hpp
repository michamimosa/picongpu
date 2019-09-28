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

#include <rmngr/scheduler/states.hpp>
#include <rmngr/resource/ioresource.hpp>
#include <rmngr/property/inherit.hpp>
#include <rmngr/property/resource.hpp>
#include <rmngr/property/label.hpp>

#include <rmngr/thread/thread_dispatcher.hpp>

namespace pmacc
{

namespace cuda_resources
{
rmngr::IOResource streams[1];
}

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

using TaskProperties = rmngr::TaskProperties<
    rmngr::ResourceProperty,
    rmngr::LabelProperty,
    PMaccProperties
>;

std::ostream& functor_backtrace(std::ostream& out);

struct EnqueuePolicy
{
    static bool is_serial(TaskProperties const & a, TaskProperties const & b)
    {
        return rmngr::ResourceUser::is_serial( a, b );
    }

    static void assert_superset(TaskProperties const & super, TaskProperties const & sub)
    {
        if(! rmngr::ResourceUser::is_superset( super, sub ))
        {
            std::stringstream stream;
            stream << "Not allowed: " << super.label << "is no superset of " << sub.label << std::endl
                   << super.label << " has access: " << std::endl
                   << (rmngr::ResourceUser)super << std::endl << std::endl
                   << sub.label << " has access: " << std::endl
	           << (rmngr::ResourceUser)sub << std::endl;
            functor_backtrace(stream);
            throw std::runtime_error(stream.str());
        }
    }
};

template < typename SchedulingGraph >
struct Scheduler
    : rmngr::StateScheduler< TaskProperties, SchedulingGraph >
{
    using TaskID = typename rmngr::TaskContainer<TaskProperties>::TaskID;
    using Job = typename SchedulingGraph::Job;

    std::mutex queue_mutex;
    std::queue< Job > main_queue;
    std::queue< Job > mpi_queue;

    bool write_graph;

    Scheduler( rmngr::TaskContainer< TaskProperties > & tasks, SchedulingGraph & graph )
        : rmngr::StateScheduler< TaskProperties, SchedulingGraph >( tasks, graph )
        , write_graph( false )
    {
        for( size_t i = 0; i < this->graph.schedule.size(); i++ )
        {
            auto & t = this->graph.schedule[i];
            if( i == 0 && this->graph.schedule.size() > 1 )
                t.set_request_hook( [this, &t]{ get_job(t, mpi_queue); });
            else
                t.set_request_hook( [this,&t]{ get_job(t, main_queue); });
        }
    }

private:
    void update_queues()
    {
        bool u1 = this->uptodate.test_and_set();
        bool u2 = this->graph.precedence_graph.test_and_set();
        if( !u1 || !u2 )
        {
            auto ready_tasks = this->update_graph();
            for( TaskID t : ready_tasks )
            {
                auto prop = this->tasks.task_properties( t );

                if( prop.mpi_task && this->graph.schedule.size() > 1 )
                    mpi_queue.push( Job{ this->tasks, t } );
                else
                    main_queue.push( Job{ this->tasks, t } );
            }
        }

        if( write_graph )
        {
            static int step=0;
            std::cout << "write step_" << step << ".dot" << std::endl;
            std::ofstream out( "step_" + std::to_string(step++) + ".dot" );
            this->graph.precedence_graph.write_dot(
                out,
                [this]( TaskID id )
                {
                    return this->tasks.task_properties(id).label;
                },
                [this]( TaskID id )
                {
                    switch( this->get_task_state(id) )
                    {
                    case rmngr::TaskState::uninitialized:
                    case rmngr::TaskState::pending:
                        return "brown";
                    case rmngr::TaskState::ready:
                        return "green";
                    case rmngr::TaskState::running:
                        return "gray";
                    case rmngr::TaskState::done:
                        return "black";
                    }
                    return "blue";
                });
        }
    }

    void get_job( typename SchedulingGraph::ThreadSchedule & thread, std::queue<Job> & queue )
    {
        std::lock_guard< std::mutex > lock( queue_mutex );

        if( queue.empty() )
            update_queues();

        if( ! queue.empty() )
        {
            auto job = queue.front();
            queue.pop();

            //std::cout << "thread["<<rmngr::thread::id<<"] RUN task \""<< this->tasks.task_properties(job.task_id).label <<"\""<<std::endl;
            thread.push( job );
        }
    }

};

} // namespace pmacc

