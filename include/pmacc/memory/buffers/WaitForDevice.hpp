
#pragma once

#include <pmacc/types.hpp>

#include <rmngr/task.hpp>
#include <pmacc/tasks/StreamTask.hpp>

#include <memory> // std::addressof()
#include <csignal>
#include <sys/mman.h>
#include <stdlib.h>
#include <unistd.h>

namespace pmacc
{


namespace waitfordevice
{  
    void* page1;
    void* page2;
    long PAGE_SIZE;
    int event_pipe[2];
    struct sigaction old_sigaction;

    void event_loop()
    {
        char v;
        while( read(event_pipe[0], &v, 1) == 1 )
        {
            std::cout << "trigger update from device" << std::endl;
	    Scheduler::getInstance().policy< rmngr::DispatchPolicy< PMaccDispatch > >().notify();
	}
    }
 
    void segv_handler(int signal, siginfo_t* siginfo, void* uap)
    {
        if( siginfo->si_addr == page1 )
	{
	    mprotect(page1, PAGE_SIZE, PROT_WRITE);
	    mprotect(page2, PAGE_SIZE, PROT_READ);
	}
	else if( siginfo->si_addr == page2 )
	{
	    mprotect(page1, PAGE_SIZE, PROT_READ);
	    mprotect(page2, PAGE_SIZE, PROT_WRITE);

	    char v = 1;
	    write(event_pipe[1], &v, 1);
	}
	else
	{
	  // A real SEGV
	  sigaction(SIGSEGV, &old_sigaction, NULL);
	}
    }

    void setup()
    {
        PAGE_SIZE = sysconf(_SC_PAGESIZE);
      /*
        static sigsegv_dispatcher dispatcher;
        sigsegv_init (&dispatcher);
        sigsegv_install_handler (&handler);
     */
        struct sigaction act;
	memset(&act, 0, sizeof(act));
	act.sa_sigaction = &segv_handler;
	act.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &act, &old_sigaction);

        pipe( event_pipe );
	std::thread event_thread(event_loop);
	event_thread.detach();

	page1 = aligned_alloc( PAGE_SIZE, PAGE_SIZE );
	page2 = aligned_alloc( PAGE_SIZE, PAGE_SIZE );

	std::cout << "pages used for WaitForDevice: p1 = " << page1 << ", p2 = " << page2 << std::endl;
	/*
	sigsegv_register (&dispatcher, page1, PAGE_SIZE, &segv_handler, &page1 );
	sigsegv_register (&dispatcher, page2, PAGE_SIZE, &segv_handler, &page2 );
	*/
        mprotect( page1, PAGE_SIZE, PROT_READ );
	mprotect( page2, PAGE_SIZE, PROT_WRITE );
    }

}


struct WaitForDeviceLabel
{
    void properties( Scheduler::Schedulable& s )
    {
        s.proto_property< GraphvizPolicy >().label = "WaitForDevice";
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

    int* init_something()
    {
        int * device_ptr;
        int value = 1;
        CUDA_CHECK(cudaMalloc( (void**) &device_ptr, sizeof(int) ));
        CUDA_CHECK(cudaMemset( device_ptr, 1, sizeof(int) ));
        return device_ptr;
    }

public:
    void run()
    {
      ///std::cout << "WAIT FOR DEVICE" << std::endl;
        static State * state_device_ptr = init_state();
	State volatile * state_host_ptr;

        static int * some_device_ptr = init_something();

        // create empty task
        auto res = Scheduler::enqueue_functor(
            [](){},
            [&state_host_ptr](Scheduler::Schedulable& s)
            {
                s.proto_property<
                    rmngr::DispatchPolicy<PMaccDispatch>
                >().dont_schedule_me = true;
                s.proto_property< GraphvizPolicy >().label = "on device";

                state_host_ptr = std::addressof( s.runtime_property<rmngr::DispatchPolicy<PMaccDispatch>>().state );
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

	// trigger sigsegv
	CUDA_CHECK(cudaMemcpyAsync(
	    waitfordevice::page1,
            some_device_ptr,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            this->getCudaStream()
        ));

	CUDA_CHECK(cudaMemcpyAsync(
	    waitfordevice::page2,
            some_device_ptr,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            this->getCudaStream()
	));

    }
};

}

