
#pragma once

#include <pmacc/types.hpp>
#include <pmacc/memory/buffers/Exchange.hpp>

namespace pmacc
{
namespace communication
{

template<
    typename T,
    size_t T_Dim
>
class SendTask
{
protected:
    Exchange<T, T_Dim> * exchange;

public:
    void properties(Scheduler::Schedulable& s)
    {
        auto& al = s.proto_property< rmngr::ResourceUserPolicy >().access_list;
        al.push_back( this->exchange->getHostBuffer().read() );
	al.push_back( this->exchange->getHostBuffer().size_resource.read() );

	s.proto_property< GraphvizPolicy >().label = "Send";
    }
};

template<
    typename T,
    size_t T_Dim
>
class ReceiveTask
{
protected:
    Exchange<T, T_Dim> * exchange;

public:
    void properties(Scheduler::Schedulable& s)
    {
        auto& al = s.proto_property< rmngr::ResourceUserPolicy >().access_list;
        al.push_back( this->exchange->getHostBuffer().write() );
        al.push_back( this->exchange->getHostBuffer().size_resource.write() );

	s.proto_property< GraphvizPolicy >().label = "Receive";
    }
};

} // namespace communication

} // namespace pmacc

