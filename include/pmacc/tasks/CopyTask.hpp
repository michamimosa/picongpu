


#pragma once

#include <pmacc/tasks/Task.hpp>

namespace pmacc
{

  namespace NEW{

template <
    typename Src,
    typename Dst
>
class CopyTask
    : public virtual Task
{
public:
    virtual ~CopyTask() {}

    virtual void properties( Scheduler::SchedulablePtr s )
    {
        auto & access = s->proto_property< rmngr::ResourceUserPolicy >().access_list;
        access.push_back( src->read() );
        access.push_back( dst->write() );
    }

protected:
    Src * src;
    Dst * dst;
};

  }

} // namespace pmacc

