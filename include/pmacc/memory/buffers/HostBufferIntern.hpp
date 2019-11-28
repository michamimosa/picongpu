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
#include <pmacc/Environment.hpp>

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
        : HostBuffer<TYPE, DIM>(size, size, redGrapes::FieldResource<DIM>{})
        , pointer(nullptr)
        , ownPointer(true)
    {
        Environment<>::get().ResourceManager().emplace_task(
            [this, size]
            {
                CUDA_CHECK(cudaMallocHost((void**)&pointer, size.productOfComponents() * sizeof (TYPE)));
                reset(false);
            },
            TaskProperties::Builder()
                .label("HostBufferIntern::HostBufferIntern()")
                .resources({
                     this->write(),
                     this->size_resource.write()
                })
        );
    }

    HostBufferIntern(HostBufferIntern& source, DataSpace<DIM> size, DataSpace<DIM> offset=DataSpace<DIM>())
        : HostBuffer<TYPE, DIM>(size, source.getPhysicalMemorySize(), source)
        , pointer(nullptr)
        , ownPointer(false)
    {
        Environment<>::get().ResourceManager().emplace_task(
            [this, &source, offset]
            {
                pointer=&(source.getDataBox()(offset));/*fix me, this is a bad way*/
                reset(true);
            },
            TaskProperties::Builder()
                .label("HostBufferIntern::HostBufferIntern(source)")
                .resources({
                    source.read(),
                    this->write(),
                    this->size_resource.write()
                })
        );
    }

    /**
     * destructor
     */
    virtual ~HostBufferIntern()
    {
        Environment<>::get().ResourceManager().emplace_task(
            [this]()
            {
                if (pointer && ownPointer)
                {
                    CUDA_CHECK_NO_EXCEPT(cudaFreeHost(pointer));
                }
            },
            TaskProperties::Builder()
                .label("HostBufferIntern::~HostBufferIntern()")
                .resources({
                    this->write(),
                    this->size_resource.write()
                })
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
        Environment<>::get().ResourceManager().emplace_task(
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
            TaskProperties::Builder()
                .label("HostBufferIntern::reset()")
                .resources({
                    this->write(),
                    this->size_resource.write()
                })
        );
    }

    void setValue(const TYPE& value)
    {
        Environment<>::get().ResourceManager().emplace_task(
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
            TaskProperties::Builder()
                .label("HostBufferIntern::setValue(" + std::to_string(value) + ")")
                .resources({
                    this->write(),
                    this->size_resource.read()
                })
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

