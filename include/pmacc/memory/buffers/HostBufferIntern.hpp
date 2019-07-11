/* Copyright 2013-2019 Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/memory/boxes/DataBoxDim1Access.hpp"
#include "pmacc/assert.hpp"

#include <pmacc/memory/buffers/CopyDeviceToHost.hpp>

namespace pmacc
{

/**
 * Internal implementation of the HostBuffer interface.
 */
template <
    class TYPE,
    std::size_t DIM
>
class HostBufferIntern : public HostBuffer<TYPE, DIM>
{
public:

    typedef typename HostBuffer<TYPE, DIM>::DataBoxType DataBoxType;

    /** constructor
     *
     * @param size extent for each dimension (in elements)
     */
    HostBufferIntern(DataSpace<DIM> size)
        : HostBuffer<TYPE, DIM>(size, size, rmngr::FieldResource<DIM>{})
        , pointer(nullptr)
        , ownPointer(true)
    {
        Scheduler::Properties prop;
        prop.policy< rmngr::ResourceUserPolicy >() += this->write();
        prop.policy< rmngr::ResourceUserPolicy >() += this->size_resource.write();
        prop.policy< GraphvizPolicy >().label = "HostBufferIntern::HostBufferIntern()";

        Scheduler::emplace_task(
            [this, size]
            {
                CUDA_CHECK(cudaMallocHost((void**)&pointer, size.productOfComponents() * sizeof (TYPE)));
                reset(false);
            },
            prop
        );
    }

    HostBufferIntern(HostBufferIntern& source, DataSpace<DIM> size, DataSpace<DIM> offset=DataSpace<DIM>())
        : HostBuffer<TYPE, DIM>(size, source.getPhysicalMemorySize(), source)
        , pointer(nullptr)
        , ownPointer(false)
    {
        Scheduler::Properties prop;
        prop.policy< rmngr::ResourceUserPolicy >() += source.read();
        prop.policy< rmngr::ResourceUserPolicy >() += this->write();
        prop.policy< rmngr::ResourceUserPolicy >() += this->size_resource.write();
        prop.policy< GraphvizPolicy >().label = "HostBufferIntern::HostBufferIntern(source)";

        Scheduler::emplace_task(
            [this, &source, offset]
            {
                pointer=&(source.getDataBox()(offset));/*fix me, this is a bad way*/
                reset(true);
            },
            prop
        );
    }

    /**
     * destructor
     */
    virtual ~HostBufferIntern()
    {
        Scheduler::Properties prop;
        prop.policy< rmngr::ResourceUserPolicy >() += this->write();
        prop.policy< rmngr::ResourceUserPolicy >() += this->size_resource.write();
        prop.policy< GraphvizPolicy >().label = "HostBufferIntern::~HostBufferIntern()";

        Scheduler::emplace_task(
            [this]()
            {
                if (pointer && ownPointer)
                {
                    CUDA_CHECK_NO_EXCEPT(cudaFreeHost(pointer));
                }
            },
            prop
        ).get();
    }

    /*! Get pointer of memory
     * @return pointer to memory
     */
    TYPE* getBasePointer()
    {
        return pointer;
    }

    TYPE* getPointer()
    {
        return pointer;
    }

    void copyFrom(DeviceBuffer<TYPE, DIM> & other)
    {
        PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        memory::buffers::copy( *this, other );
    }

    void reset(bool preserveData = true)
    {
        Scheduler::Properties prop;
        prop.policy< rmngr::ResourceUserPolicy >() += this->write();
        prop.policy< rmngr::ResourceUserPolicy >() += this->size_resource.write();
        prop.policy< GraphvizPolicy >().label = "HostBufferIntern::reset()";

        Scheduler::emplace_task(
            [this, preserveData]
            {
                this->setCurrentSize(this->getDataSpace().productOfComponents());
                if (!preserveData)
                {
                    /* if it is a pointer out of other memory we can not assume that
                     * that the physical memory is contiguous
                     */
                    if(ownPointer)
                        memset(pointer, 0, this->getDataSpace().productOfComponents() * sizeof (TYPE));
                    else
                    {
                        TYPE value;
                        /* using `uint8_t` for byte-wise looping through tmp var value of `TYPE` */
                        uint8_t* valuePtr = (uint8_t*)&value;
                        for( size_t b = 0; b < sizeof(TYPE); ++b)
                        {
                            valuePtr[b] = static_cast<uint8_t>(0);
                        }
                        /* set value with zero-ed `TYPE` */
                        setValue(value);
                    }
                }
            },
            prop
        );
    }

    void setValue(const TYPE& value)
    {
        Scheduler::Properties prop;
        prop.policy< rmngr::ResourceUserPolicy >() += this->write();
        prop.policy< rmngr::ResourceUserPolicy >() += this->size_resource.read();
        prop.policy< GraphvizPolicy >().label = "HostBufferIntern::setValue()";

        Scheduler::emplace_task(
            [this, value]
            {
                int64_t current_size = static_cast< int64_t >(this->getCurrentSize());
                DataBoxDim1Access< DataBoxType > d1Box(
                    this->getDataBox(),
                    this->getDataSpace()
                );

                #pragma omp parallel for
                for (int64_t i = 0; i < current_size; i++)
                {
                    d1Box[i] = value;
                }
            },
            prop
        );
    }

    DataBoxType getDataBox()
    {
        return DataBoxType(
                   PitchedBox<TYPE, DIM>(
                       pointer,
                       DataSpace<DIM>(),
                       this->getPhysicalMemorySize(),
                       this->getPhysicalMemorySize()[0] * sizeof (TYPE)
                   )
               );
    }

private:
    TYPE* pointer;
    bool ownPointer;
};

} // namespace pmacc

