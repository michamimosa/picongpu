
#pragma once

namespace pmacc
{
  namespace NEW{

class StreamTask
{
public:
    virtual ~StreamTask() {};

    cudaStream_t getCudaStream() const
    {
        return 0;
    }

    virtual void properties( Scheduler::SchedulablePtr s )
    {
        s->proto_property< rmngr::ResourceUserPolicy >().access_list.push_back(
            this->getStreamResource().write()
        );
    }

    rmngr::IOResource getStreamResource()
    {
        static rmngr::IOResource stream_resource;
        return stream_resource;
    }
};

  }

} // namespace pmacc

