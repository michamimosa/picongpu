
#pragma once

#include <vector>
#include <forward_list>
#include <pmacc/Environment.hpp>
#include <mpi.h>


#include <thread>
#include <chrono>

namespace pmacc
{
namespace communication
{

auto task_mpi_test( MPI_Request * request, MPI_Status * status )
{
    return Environment<>::get().ResourceManager().emplace_task(
        [request, status]
        {
            int flag = 0;
            MPI_CHECK(MPI_Test(request, &flag, status));
            return bool(flag);
        },
        TaskProperties::Builder()
            .label("mpi_test(request = " + std::to_string(size_t(request)) + ")")
            .mpi_task()
    );
}

struct MPIRequestPool
{
    static MPIRequestPool & get()
    {
        static MPIRequestPool instance;
        return instance;
    }

    using EventID = typename rmngr::SchedulingGraph<pmacc::TaskProperties>::EventID;

    std::recursive_mutex mutex;
    std::map<MPI_Request*, std::pair<EventID, std::shared_ptr<MPI_Status>>> requests;

    void poll()
    {
        std::vector<std::tuple<MPI_Request*, EventID, std::shared_ptr<MPI_Status>>> r;
        {
            std::lock_guard<std::recursive_mutex> lock(mutex);
            r.reserve(requests.size());
            for( auto it = requests.begin(); it != requests.end(); ++it )
                r.emplace_back(it->first, it->second.first, it->second.second);
        }

        for( auto request : r )
        {
            std::future<bool> res = task_mpi_test(std::get<0>(request), std::get<2>(request).get());
            if( res.get() )
            {
                {
                    std::lock_guard<std::recursive_mutex> lock(mutex);
                    requests.erase( std::get<0>(request) );
                }

                Environment<>::get()
                    .ResourceManager()
                    .getScheduler()
                    .graph.finish_event( std::get<1>(request) );
                Environment<>::get().ResourceManager().getScheduler().notify();
            }
        }
    }

    void insert( MPI_Request * request, EventID event, std::shared_ptr<MPI_Status> status )
    {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        requests[request].first = event;
        requests[request].second = status;
    }
};

auto task_mpi_wait( MPI_Request * request )
{
    auto status = std::make_shared< MPI_Status >();
    rmngr::IOResource status_resource;

    Environment<>::get().ResourceManager().emplace_task(
        [request, status]
        {
            auto task_id = *Environment<>::get().ResourceManager().getScheduler().graph.get_current_task();
            auto event_id = Environment<>::get().ResourceManager().getScheduler().graph.add_post_dependency( task_id );
            MPIRequestPool::get().insert( request, event_id, status );
        },
        TaskProperties::Builder()
            .label("mpi_wait(request = " + std::to_string(size_t(request)) + ")")
            .resources({ status_resource.write() })
    );

    return std::make_pair(status_resource, status);
}

} // namespace communication

} // namespace pmacc

