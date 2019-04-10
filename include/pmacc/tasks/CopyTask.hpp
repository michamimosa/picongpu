
#pragma once

#include <pmacc/types.hpp>

namespace pmacc
{

  namespace NEW{

template <
    typename Src,
    typename Dst
>
class CopyTask
{
public:
    void properties( Scheduler::Schedulable& s )
    {
        auto & access = s.proto_property< rmngr::ResourceUserPolicy >().access_list;
        access.push_back( src->read() );
        access.push_back( src->size_resource.write() );

        access.push_back( dst->write() );
        access.push_back( dst->size_resource.write() );
    }

protected:
    Src * src;
    Dst * dst;
};

  }

} // namespace pmacc

