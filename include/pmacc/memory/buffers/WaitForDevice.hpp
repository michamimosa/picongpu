
#pragma once

#include <pmacc/types.hpp>

#include <rmngr/task.hpp>
#include <pmacc/tasks/StreamTask.hpp>

#include <csignal>
#include <sys/mman.h>
#include <stdlib.h>

namespace pmacc
{

struct WaitForDeviceLabel
{
    void properties( Scheduler::SchedulablePtr s )
    {
        s->proto_property< GraphvizPolicy >().label = "WaitForDevice";
    }
};

class TaskWaitForDevice
    : public rmngr::Task<
        TaskWaitForDevice,
            boost::mpl::vector<
                NEW::StreamTask,
                WaitForDeviceLabel
            >
        >
{
public:
    TaskWaitForDevice() {}

private:
    using State = rmngr::DispatchPolicy<PMaccDispatch>::RuntimeProperty::State;

    State* init_state()
    {
        State * device_ptr;
        State value = State::done;
        CUDA_CHECK(cudaMalloc( (void**) &device_ptr, sizeof(State) ));
        CUDA_CHECK(cudaMemset( device_ptr, 0, sizeof(State) ));
        CUDA_CHECK(cudaMemset( device_ptr, value, 1 ));
        return device_ptr;
    }

    std::atomic_flag* init_clear_flag()
    {
        std::atomic_flag * device_ptr;
        std::atomic_flag value = ATOMIC_FLAG_INIT;
        CUDA_CHECK(cudaMalloc( (void**) &device_ptr, sizeof(std::atomic_flag) ));
        CUDA_CHECK(cudaMemset( device_ptr, *reinterpret_cast<int*>(&value), sizeof(std::atomic_flag) ));
        return device_ptr;
    }

public:
    void run()
    {
        static State * state_device_ptr = init_state();
        static std::atomic_flag * flag_device_ptr = init_clear_flag();

        State volatile * state_host_ptr;
        std::atomic_flag volatile * flag_host_ptr = &Scheduler::getInstance().uptodate;

        // create empty task
        auto res = Scheduler::enqueue_functor(
            [](){},
            [&state_host_ptr](Scheduler::SchedulablePtr s)
            {
                s->proto_property<
                    rmngr::DispatchPolicy<PMaccDispatch>
                >().dont_schedule_me = true;
                s->proto_property< GraphvizPolicy >().label = "on device";

                state_host_ptr = & ( s->runtime_property<rmngr::DispatchPolicy<PMaccDispatch>>().state );
            }
        );

        // set state of dummy task to done
        CUDA_CHECK(cudaMemcpyAsync(
            const_cast<State*>(state_host_ptr),
            state_device_ptr,
            sizeof(State),
            cudaMemcpyDeviceToHost,
            this->getCudaStream()
        ));

        // mark graph dirty
        CUDA_CHECK(cudaMemcpyAsync(
            const_cast<std::atomic_flag*>(flag_host_ptr),
            flag_device_ptr,
            sizeof(std::atomic_flag),
            cudaMemcpyDeviceToHost,
            this->getCudaStream()
        ));
    }
};

}

