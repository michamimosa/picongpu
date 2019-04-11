
#pragma once

#include <pmacc/types.hpp>

namespace pmacc
{
namespace communication
{

/**
 * Task type for MPI-operations
 * A MPITask will always get scheduled to the main thread
 */
class MPITask
{
public:
    void properties( Scheduler::Schedulable& s )
    {
        s.proto_property< rmngr::DispatchPolicy<PMaccDispatch> >()
          .main_thread = true;
    }
};

} // namespace communication

} // namespace pmacc

