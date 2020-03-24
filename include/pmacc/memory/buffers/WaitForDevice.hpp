
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

    using EventID = typename std::remove_reference<decltype(Environment<>::get().ResourceManager())>::type::EventID;

    struct KernelNotify
    {
        template <typename T_Acc>
        DINLINE void operator() (
             T_Acc const & acc,
             EventID * ptr,
             EventID event_id
        ) const
        {
            *ptr = event_id;
        }
    };

    void event_loop()
    {
        EventID id;
        while( read(event_pipe[0], &id, sizeof(EventID)) == sizeof(EventID) )
            Environment<>::get().ResourceManager().reach_event( id );
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

	    EventID id = *((EventID*)page1);
	    write(event_pipe[1], &id, sizeof(EventID));
	}
	else
	{
            functor_backtrace( std::cerr );
            throw std::runtime_error("Segmentation Fault !");

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

    
    EventID* init_event_id()
    {
        EventID * device_ptr;
        CUDA_CHECK(cudaMalloc( (void**) &device_ptr, sizeof(EventID) ));
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
    return Environment<>::get().ResourceManager().emplace_task(
        [cuda_stream]
        {
            static waitfordevice::EventID * event_id_device_ptr = waitfordevice::init_event_id();
            static int * some_device_ptr = waitfordevice::init_something();

            auto event_id = *Environment<>::get().ResourceManager().create_event();
            CUPLA_KERNEL
                ( waitfordevice::KernelNotify )
                ( 1, 1, 0, cuda_stream )
                ( event_id_device_ptr, event_id );

            CUDA_CHECK(cudaMemcpyAsync(
                waitfordevice::page1,
                event_id_device_ptr,
                sizeof(waitfordevice::EventID),
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
        TaskProperties::Builder()
            .label("task_synchronize_stream(" + std::to_string(0) + ")")
            .resources({
                cuda_resources::streams[0].write()
            })
    );
}

} // namespace pmacc

