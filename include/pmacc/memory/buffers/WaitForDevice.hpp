
#pragma once

#include <pmacc/types.hpp>

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
            Scheduler::getInstance().uptodate.clear();
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
            functor_backtrace(std::cerr);

            // A real SEGV
            sigaction(SIGSEGV, &old_sigaction, NULL);
	}
    }

    void setup()
    {
        PAGE_SIZE = sysconf(_SC_PAGESIZE);

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

        mprotect( page1, PAGE_SIZE, PROT_READ );
        mprotect( page2, PAGE_SIZE, PROT_WRITE );
    }

    using State = rmngr::DispatchPolicy<PMaccDispatch>::Property::State;

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
} // namespace waitfordevice

auto task_synchronize_stream( cudaStream_t cuda_stream )
{
    Scheduler::Properties prop;
    prop.policy< rmngr::ResourceUserPolicy >() += cuda_resources::streams[0].write();
    prop.policy< GraphvizPolicy >().label = "task_synchronize_stream()";

    return Scheduler::emplace_task(
        [cuda_stream]
        {
            static waitfordevice::State * state_device_ptr = waitfordevice::init_state();
            waitfordevice::State volatile * state_host_ptr;

            static int * some_device_ptr = waitfordevice::init_something();

            // create empty task
            Scheduler::Properties prop;
            prop.policy< rmngr::DispatchPolicy<PMaccDispatch> >().job_selector_prop.dont_schedule_me = true;
            prop.policy< GraphvizPolicy >().label = "on device";

            auto task = new Scheduler::FunctorTask<std::function<void(void)>>( Scheduler::getInstance(), []{}, prop );
            state_host_ptr = std::addressof( task->property<rmngr::DispatchPolicy<PMaccDispatch>>().state );

            Scheduler::getInstance().push( task );

            // set state of dummy task to done
            CUDA_CHECK(cudaMemcpyAsync(
                const_cast<waitfordevice::State*>(state_host_ptr),
                state_device_ptr,
                sizeof(waitfordevice::State),
                cudaMemcpyDeviceToHost,
                cuda_stream
            ));

            // trigger sigsegv
            CUDA_CHECK(cudaMemcpyAsync(
                waitfordevice::page1,
                some_device_ptr,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                cuda_stream
            ));

            CUDA_CHECK(cudaMemcpyAsync(
	        waitfordevice::page2,
                some_device_ptr,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                cuda_stream
	    ));
        },
        prop
    );
}

} // namespace pmacc

