/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund, Michael Sippel
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/assert.hpp"

#include <pmacc/memory/buffers/SetValueOnDevice.hpp>

#include <pmacc/exec/kernelEvents.hpp>
#include "pmacc/nvidia/gpuEntryFunction.hpp"

#include <pmacc/Environment.hpp>

namespace pmacc
{
namespace mem
{

/**
 * Internal device buffer implementation.
 */
template <class TYPE, std::size_t DIM, typename T_DataAccessPolicy>
class DeviceBufferIntern : public DeviceBuffer<TYPE, DIM, T_DataAccessPolicy>
{
public:
    typedef typename DeviceBuffer<TYPE, DIM, T_DataAccessPolicy>::DataBoxType DataBoxType;

    using Item = TYPE;
    static constexpr std::size_t dim = DIM;

    /*! create device buffer
     * @param size extent for each dimension (in elements)
     * @param sizeOnDevice memory with the current size of the grid is stored on device
     * @param useVectorAsBase use a vector as base of the array (is not lined pitched)
     *                      if true size on device is atomaticly set to false
     */
    void init(DataSpace<DIM> size,
              bool sizeOnDevice = false,
              bool useVectorAsBase = false)
    {
        this->DeviceBuffer<TYPE, DIM, T_DataAccessPolicy>::init( size, size );

        Environment<>::task(
            [size, sizeOnDevice, useVectorAsBase]( auto x )
            {
                auto obj = std::static_pointer_cast< DeviceBufferIntern >( x );
                obj->sizeOnDevice = sizeOnDevice;
                obj->useOtherMemory = false;

                // create size on device before any use of setCurrentSize
                if (useVectorAsBase)
                {
                    obj->sizeOnDevice = false;
                    obj->createSizeOnDevice(obj->sizeOnDevice);
                    obj->createFakeData();
                    obj->data1D = true;
                }
                else
                {
                    obj->createSizeOnDevice(obj->sizeOnDevice);
                    obj->createData();
                    obj->data1D = false;
                }
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::DeviceBufferIntern()")
                .resources({
                    Environment<>::get().cuda_stream()
                }),
            this->buffer_resource().write()
        );
    }

    void init(DeviceBuffer<TYPE, DIM, T_DataAccessPolicy>& source,
              DataSpace<DIM> size,
              DataSpace<DIM> offset,
              bool sizeOnDevice = false)
    {
        this->DeviceBuffer<TYPE, DIM, T_DataAccessPolicy>::init( size, source.getPhysicalMemorySize(), source );
        Environment<>::task(
            [&source, offset, sizeOnDevice]( auto x )
            {
                auto obj = std::static_pointer_cast< DeviceBufferIntern >( x );
                obj->sizeOnDevice = sizeOnDevice;
                obj->useOtherMemory = true;
                obj->offset = offset + source.getOffset();
                obj->data = source.getCudaPitched();
                obj->createSizeOnDevice(obj->sizeOnDevice);
                obj->data1D = false;
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::DeviceBufferIntern(source)")
                .resources({
                    Environment<>::get().cuda_stream()
                }),
            this->buffer_resource().write()
        );
    }

    virtual ~DeviceBufferIntern()
    {
        if (sizeOnDevice)
        {
            CUDA_CHECK_NO_EXCEPT(cudaFree(sizeOnDevicePtr));
        }

        if (!useOtherMemory)
        {
            CUDA_CHECK_NO_EXCEPT(cudaFree(data.ptr));
        }
    }

    void reset( bool preserveData = true )
    {
        Environment<>::task(
            [preserveData]( auto x )
            {
                auto obj = std::static_pointer_cast< DeviceBufferIntern >( x );
                obj->set_size(obj->Buffer<Item, dim, T_DataAccessPolicy>::getDataSpace().productOfComponents());

                if (!preserveData)
                {
                    Item value;

                    /* using `uint8_t` for byte-wise looping through tmp var value of `TYPE` */
                    uint8_t* valuePtr = reinterpret_cast<uint8_t*>(&value);
                    for( size_t b = 0; b < sizeof(Item); ++b)
                    {
                        valuePtr[b] = static_cast<uint8_t>(0);
                    }

                    /* set value with zero-ed `TYPE` */
                    obj->fill( value );
                }
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::reset()"),
            this->buffer_resource().write()
        );
    }

    DataBoxType getDataBox()
    {
        return DataBoxType(
                   PitchedBox<TYPE, DIM >(
                       (TYPE*) data.ptr,
                       offset,
                       this->getPhysicalMemorySize(),
                       data.pitch
                   )
               );
    }

    Item * getPointer() const
    {
        if (DIM == DIM1)
        {
            return (TYPE*) (data.ptr) + this->offset[0];
        }
        else if (DIM == DIM2)
        {
            return (TYPE*) ((char*) data.ptr + this->offset[1] * this->data.pitch) + this->offset[0];
        }
        else
        {
            const size_t offsetY = this->offset[1] * this->data.pitch;
            const size_t sizePlaneXY = this->getPhysicalMemorySize()[1] * this->data.pitch;
            return (TYPE*) ((char*) data.ptr + this->offset[2] * sizePlaneXY + offsetY) + this->offset[0];
        }
    }

    DataSpace<DIM> getOffset() const
    {
        return offset;
    }

    bool hasCurrentSizeOnDevice() const
    {
        return sizeOnDevice;
    }

    size_t* getCurrentSizeOnDevicePointer()
    {
        if (!sizeOnDevice)
        {
            throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
        }
        return sizeOnDevicePtr;
    }

    size_t* getCurrentSizeHostSidePointer()
    {
        return this->current_size;
    }

    TYPE* getBasePointer() const
    {
        return (TYPE*) data.ptr;
    }

    /*! Get current size of any dimension
     * @return count of current elements per dimension
     */
    virtual std::size_t get_size()
    {
        return Environment<>::task(
            []( auto x, auto cuda_stream )
            {
                auto obj = std::static_pointer_cast< DeviceBufferIntern >( x );
                if (obj->sizeOnDevice)
                {
                    CUDA_CHECK(cudaMemcpyAsync((void*) obj->getCurrentSizeHostSidePointer(),
                                               obj->getCurrentSizeOnDevicePointer(),
                                               sizeof (size_t),
                                               cudaMemcpyDeviceToHost,
                                               cuda_stream));
                    cuda_stream->sync().get();
                }

                return obj->DeviceBuffer<Item, dim, T_DataAccessPolicy>::get_size();
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::get_size()"),
            this->buffer_resource().size().read(),
            Environment<>::get().cuda_stream()
        ).get();
    }

    struct KernelSetValueOnDeviceMemory
    {
        template< typename T_Acc >
        DINLINE void operator()(T_Acc const &, size_t* pointer, size_t const size) const
        {
            *pointer = size;
        }
    };

    virtual void set_size( std::size_t const new_size )
    {
        Environment<>::task(
            [new_size]( auto x, auto cuda_stream ) {
                auto obj = std::static_pointer_cast< DeviceBufferIntern >( x );
                if (obj->sizeOnDevice)
                {
                    obj->Buffer< Item, dim >::set_size( new_size );
                    CUPLA_KERNEL( KernelSetValueOnDeviceMemory )(
                        1,
                        1,
                        0,
                        cuda_stream
                    )(
                        obj->getCurrentSizeOnDevicePointer(),
                        new_size
                    );
                }
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::set_size()"),
            this->buffer_resource().size().write(),
            Environment<>::get().cuda_stream()
        );
    }
    /*
    void copyFrom(buffer::ReadGuard<HostBuffer<TYPE, DIM>> other)
    {
        PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other->getCurrentDataSpace()));
        memory::buffers::copy( *this, other );
    }

    void copyFrom(buffer::ReadGuard<DeviceBuffer<TYPE, DIM>> other)
    {
        PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other->getCurrentDataSpace()));
        memory::buffers::copy( *this, other );
    }
    */
    cudaPitchedPtr const getCudaPitched() const
    {
        return data;
    }

    size_t getPitch() const
    {
        return data.pitch;
    }

    virtual void fill( Item const & value )
    {
        enum
        {
            isSmall = (sizeof(Item) <= 128)
        }; //if we use const variable the compiler create warnings

        auto res = BufferResource< DeviceBuffer< Item, dim, T_DataAccessPolicy > >( this->template shared_from_base<DeviceBuffer<Item, dim, T_DataAccessPolicy>>() );

        if( isSmall )
            buffer::device_set_value_small( res.write(), value );
        else
            buffer::device_set_value_big( res.write(), value );
    }

private:
    std::shared_ptr< DeviceBufferIntern< Item, dim, T_DataAccessPolicy > > shared_from_this()
    {
        return this->template shared_from_base< DeviceBufferIntern<Item, dim, T_DataAccessPolicy> >();
    }

    /*! create native array with pitched lines
     */
    void createData()
    {
        Environment<>::task(
            []( auto x )
            {
                auto obj = std::static_pointer_cast< DeviceBufferIntern >( x );
                obj->data.ptr = nullptr;
                obj->data.pitch = 1;
                obj->data.xsize = obj->getDataSpace()[0] * sizeof(Item);
                obj->data.ysize = 1;

                if (DIM == DIM1)
                {
                    log<ggLog::MEMORY >("Create device 1D data: %1% MiB") % (obj->data.xsize / 1024 / 1024);
                    CUDA_CHECK(cudaMallocPitch(&obj->data.ptr, &obj->data.pitch, obj->data.xsize, 1));
                }
                if (DIM == DIM2)
                {
                    obj->data.ysize = obj->getDataSpace()[1];
                    log<ggLog::MEMORY >("Create device 2D data: %1% MiB") % (obj->data.xsize * obj->data.ysize / 1024 / 1024);
                    CUDA_CHECK(cudaMallocPitch(
                        &obj->data.ptr,
                        &obj->data.pitch,
                        obj->data.xsize,
                        obj->data.ysize));
                }
                if (DIM == DIM3)
                {
                    cudaExtent extent;
                    extent.width = obj->getDataSpace()[0] * sizeof (TYPE);
                    extent.height = obj->getDataSpace()[1];
                    extent.depth = obj->getDataSpace()[2];

                    log<ggLog::MEMORY >("Create device 3D data: %1% MiB") % (obj->getDataSpace().productOfComponents() * sizeof(Item) / 1024 / 1024);
                    CUDA_CHECK(cudaMalloc3D(&obj->data, extent));
                }

                obj->reset( false );
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::createData()"),
            this->buffer_resource().write()
        );
    }

    /*!create 1D, 2D, 3D Array which use only a vector as base
     */
    void createFakeData()
    {
        Environment<>::task(
            []( auto x )
            {
                auto obj = std::static_pointer_cast< DeviceBufferIntern >( x );
                obj->data.ptr = nullptr;
                obj->data.pitch = 1;
                obj->data.xsize = obj->getDataSpace()[0] * sizeof(Item);
                obj->data.ysize = 1;

                log<ggLog::MEMORY >("Create device fake data: %1% MiB") % (obj->getDataSpace().productOfComponents() * sizeof(Item) / 1024 / 1024);
                CUDA_CHECK(cudaMallocPitch(&obj->data.ptr, &obj->data.pitch, obj->getDataSpace().productOfComponents() * sizeof(Item), 1));

                //fake the pitch, thus we can use this 1D Buffer as 2D or 3D
                obj->data.pitch = obj->getDataSpace()[0] * sizeof(Item);

                if (DIM > DIM1)
                {
                    obj->data.ysize = obj->getDataSpace()[1];
                }

                obj->reset( false );
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::createFakeData()"),
            this->buffer_resource().write()
        );
    }

    void createSizeOnDevice(bool sizeOnDevice)
    {
        Environment<>::task(
            [sizeOnDevice]( auto x )
            {
                auto obj = std::static_pointer_cast< DeviceBufferIntern >( x );
                obj->sizeOnDevicePtr = nullptr;

                if (obj->sizeOnDevice)
                {
                    CUDA_CHECK(cudaMalloc((void**)&obj->sizeOnDevicePtr, sizeof (size_t)));
                }

                obj->set_size( obj->getDataSpace().productOfComponents() );
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::createSizeOnDevice()"),
            this->buffer_resource().size().write()
        );
    }

private:
    DataSpace<DIM> offset;

    bool sizeOnDevice;
    size_t* sizeOnDevicePtr;
    cuplaPitchedPtr data;
    bool useOtherMemory;
};

} // namespace mem

} // namespace pmacc

