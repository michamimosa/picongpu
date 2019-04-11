/* Copyright 2013-2018 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <iomanip>
#include <pmacc/types.hpp>
#include <pmacc/memory/buffers/CopyTask.hpp>
#include <pmacc/tasks/StreamTask.hpp>
#include <rmngr/task.hpp>

#include "WaitForDevice.hpp"

namespace pmacc
{

template < typename T, unsigned T_Dim >
class HostBuffer;

template < typename T, unsigned T_Dim >
class DeviceBuffer;

namespace memory
{
namespace buffers
{

struct LabelDeviceToHost
{
    void properties(Scheduler::Schedulable& s)
    {
        s.proto_property< GraphvizPolicy >().label = "CopyDeviceToHost()";
    }
};

template <
    typename Impl,
    typename T,
    size_t T_Dim
>
class TaskCopyDeviceToHostBase
    : public rmngr::Task<
          Impl,
          boost::mpl::vector<
	    NEW::StreamTask,
              CopyTask<
                  DeviceBuffer<T, T_Dim>,
                  HostBuffer<T, T_Dim>
              >,
              LabelDeviceToHost
          >
      >

{
public:
    TaskCopyDeviceToHostBase(
        DeviceBuffer<T, T_Dim> & src,
        HostBuffer<T, T_Dim> & dst
    )
    {
        this->src = &src;
        this->dst = &dst;
    }

    void run()
    {
        size_t current_size = this->src->getCurrentSize();

        this->dst->setCurrentSize(current_size);
        DataSpace<T_Dim> devCurrentSize = this->src->getCurrentDataSpace(current_size);

        if (this->src->is1D() && this->dst->is1D())
            fastCopy(this->src->getPointer(), this->dst->getPointer(), devCurrentSize.productOfComponents());
        else
            copy(devCurrentSize);

        TaskWaitForDevice::create( Scheduler::getInstance() );
    }

protected:
    virtual void copy(DataSpace<T_Dim> &devCurrentSize) = 0;

    void fastCopy(T* src, T* dst, size_t size)
    {
        CUDA_CHECK(cudaMemcpyAsync(dst,
                                   src,
                                   size * sizeof (T),
                                   cudaMemcpyDeviceToHost,
                                   this->getCudaStream()));
    }
};

template < typename T, unsigned T_Dim>
class TaskCopyDeviceToHost;

template < typename T >
class TaskCopyDeviceToHost< T, DIM1 >
    : public TaskCopyDeviceToHostBase<
          TaskCopyDeviceToHost< T, DIM1 >,
          T,
          DIM1
      >
{
public:
    TaskCopyDeviceToHost( DeviceBuffer<T, DIM1>& src, HostBuffer<T, DIM1>& dst)
      : TaskCopyDeviceToHostBase<TaskCopyDeviceToHost<T, DIM1>, T, DIM1>(src, dst)
    {}

private:
    virtual void copy(DataSpace<DIM1> & devCurrentSize)
    {
        CUDA_CHECK(cudaMemcpyAsync(this->dst->getBasePointer(),
                                   this->src->getPointer(),
                                   devCurrentSize[0] * sizeof (T),
                                   cudaMemcpyDeviceToHost,
                                   this->getCudaStream()));
    }
};

template < typename T >
class TaskCopyDeviceToHost< T, DIM2 >
    : public TaskCopyDeviceToHostBase<
          TaskCopyDeviceToHost< T, DIM2 >,
          T,
          DIM2
      >
{
public:
    TaskCopyDeviceToHost(DeviceBuffer<T, DIM2> & src, HostBuffer<T, DIM2> & dst)
      : TaskCopyDeviceToHostBase<TaskCopyDeviceToHost<T, DIM2>, T, DIM2>(src, dst)
    {}

private:
    virtual void copy(DataSpace<DIM2> &devCurrentSize)
    {
        CUDA_CHECK(cudaMemcpy2DAsync(this->dst->getBasePointer(),
                                     this->dst->getDataSpace()[0] * sizeof (T), /*this is pitch*/
                                     this->src->getPointer(),
                                     this->src->getPitch(), /*this is pitch*/
                                     devCurrentSize[0] * sizeof (T),
                                     devCurrentSize[1],
                                     cudaMemcpyDeviceToHost,
                                     this->getCudaStream()));
    }
};

template <class TYPE>
class TaskCopyDeviceToHost< TYPE, DIM3 >
    : public TaskCopyDeviceToHostBase<
          TaskCopyDeviceToHost< TYPE, DIM3 >,
          TYPE,
          DIM3
      >
{
public:
    TaskCopyDeviceToHost( DeviceBuffer<TYPE, DIM3>& src, HostBuffer<TYPE, DIM3>& dst )
      : TaskCopyDeviceToHostBase<TaskCopyDeviceToHost<TYPE, DIM3
                                                      >, TYPE, DIM3>(src, dst)
    {}

private:
    virtual void copy(DataSpace<DIM3> &devCurrentSize)
    {
        cudaPitchedPtr hostPtr;
        hostPtr.pitch = this->dst->getDataSpace()[0] * sizeof (TYPE);
        hostPtr.ptr = this->dst->getBasePointer();
        hostPtr.xsize = this->dst->getDataSpace()[0] * sizeof (TYPE);
        hostPtr.ysize = this->dst->getDataSpace()[1];

        cudaMemcpy3DParms params;
        params.srcArray = nullptr;
        params.srcPos = make_cudaPos(this->src->getOffset()[0] * sizeof (TYPE),
                                     this->src->getOffset()[1],
                                     this->src->getOffset()[2]);
        params.srcPtr = this->src->getCudaPitched();

        params.dstArray = nullptr;
        params.dstPos = make_cudaPos(0, 0, 0);
        params.dstPtr = hostPtr;

        params.extent = make_cudaExtent(
                            devCurrentSize[0] * sizeof (TYPE),
                            devCurrentSize[1],
                            devCurrentSize[2]);
        params.kind = cudaMemcpyDeviceToHost;

        CUDA_CHECK(cudaMemcpy3DAsync(&params, this->getCudaStream()))
    }
};

} // namespace buffers

} // namespace memory

} // namespace pmacc

