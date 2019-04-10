
#pragma once

#include <pmacc/types.hpp>

namespace pmacc
{
  namespace NEW{

class StreamTask
{
public:
    cudaStream_t getCudaStream() const
    {
        return 0;
    }

    void properties( Scheduler::Schedulable& s ) const
    {
        s.proto_property< rmngr::ResourceUserPolicy >().access_list.push_back(
            this->getStreamResource().write()
        );
    }

    rmngr::IOResource getStreamResource() const
    {
        static rmngr::IOResource stream_resource;
        return stream_resource;
    }
};

  }

} // namespace pmacc

