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

#include <pmacc/memory/buffers/WaitForDevice.hpp>
#include <pmacc/memory/buffers/SetValueOnDevice.hpp>

#include <pmacc/memory/buffers/CopyHostToDevice.hpp>
#include <pmacc/memory/buffers/CopyDeviceToDevice.hpp>

#include <pmacc/exec/kernelEvents.hpp>
#include "pmacc/nvidia/gpuEntryFunction.hpp"

#include <pmacc/Environment.hpp>

namespace pmacc
{

/**
 * Internal device buffer implementation.
 */
template <class TYPE, std::size_t DIM>
class DeviceBufferIntern : public DeviceBuffer<TYPE, DIM>
{
public:

    typedef typename DeviceBuffer<TYPE, DIM>::DataBoxType DataBoxType;

    /*! create device buffer
     * @param size extent for each dimension (in elements)
     * @param sizeOnDevice memory with the current size of the grid is stored on device
     * @param useVectorAsBase use a vector as base of the array (is not lined pitched)
     *                      if true size on device is atomaticly set to false
     */
    DeviceBufferIntern(DataSpace<DIM> size,
                       bool sizeOnDevice = false,
                       bool useVectorAsBase = false)
        : DeviceBuffer<TYPE, DIM>(size, size, redGrapes::FieldResource<DIM>{})
        , sizeOnDevice(sizeOnDevice)
        , useOtherMemory(false)
        , offset(DataSpace<DIM>())
    {
        Environment<>::task(
            [obj=shared_from_this(), size, useVectorAsBase]
            {
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
                    access_size_write( rg::access::Field<dim>(rg::access::IO::write) )
                    cuda_resources::streams[0].write()
                })
        );
    }

    DeviceBufferIntern(DeviceBuffer<TYPE, DIM>& source,
                       DataSpace<DIM> size,
                       DataSpace<DIM> offset,
                       bool sizeOnDevice = false)
        : DeviceBuffer<TYPE, DIM>(size, source.getPhysicalMemorySize(), source)
        , sizeOnDevice(sizeOnDevice)
        , useOtherMemory(true)
    {
        Environment<>::task(
            [obj=shared_from_this(), &source, offset]
            {
                obj->offset = offset + source.getOffset();
                obj->data = source.getCudaPitched();
                obj->createSizeOnDevice(obj->sizeOnDevice);
                obj->data1D = false;
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::DeviceBufferIntern(source)")
                .resources({
                    access_size_write( rg::access::Field<dim>(rg::access::IO::write) ),
                    cuda_resources::streams[0].write()
                })
        );
    }

    virtual ~DeviceBufferIntern()
    {
        Environment<>::task(
            [obj=shared_from_this()]
            {
                if (obj->sizeOnDevice)
                {
                    CUDA_CHECK_NO_EXCEPT(cudaFree(obj->sizeOnDevicePtr));
                }
                if (!obj->useOtherMemory)
                {
                    CUDA_CHECK_NO_EXCEPT(cudaFree(obj->data.ptr));
                }
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::~DeviceBufferIntern()")
                .resources({
                    access_size_write( rg::access::Field<dim>(rg::access::IO::write) )
                })
        );
    }

    void reset(bool preserveData = true)
    {
        Environment<>::task(
            [obj=shared_from_this(), preserveData]
            {
                obj->set_size(Buffer<Item, dim>::getDataSpace().productOfComponents());

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
                    fill(value);
                }
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::reset()")
                .resources({
                    access_size_write( rg::access::Field<dim>(rg::access::IO::write) ),
                    cuda_resources::streams[0].write()
                })
        );
    }

    DataBoxType getDataBox()
    {
        return DataBoxType(
                   PitchedBox<TYPE, DIM >( (TYPE*) data.ptr, offset, this->getPhysicalMemorySize(), data.pitch )
               );
    }

    TYPE* getPointer()
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

    TYPE* getBasePointer()
    {
        return (TYPE*) data.ptr;
    }

    /*! Get current size of any dimension
     * @return count of current elements per dimension
     */
    virtual auto get_size()
    {
        return Environment<>::task(
            [obj=shared_from_this()]
            {
                if (obj->sizeOnDevice)
                {
                    cudaStream_t cuda_stream = 0;

                    CUDA_CHECK(cudaMemcpyAsync((void*) obj->getCurrentSizeHostSidePointer(),
                                               obj->getCurrentSizeOnDevicePointer(),
                                               sizeof (size_t),
                                               cudaMemcpyDeviceToHost,
                                               cuda_stream));

                    task_synchronize_stream(0);
                }

                Environment<>::get()
                    .ResourceManager()
                    .update_properties(
                        TaskProperties::Patch::Builder()
                            .remove_resources({ access_size_write() })
                    );

                return DeviceBuffer<Item, dim>::get_size();
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::get_size()")
                .resources({
                    access_size_read(),
                    access_size_write()
                    cuda_resources::streams[0].write()
                })
        );
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
           [obj=shared_from_this(), new_size] {
                Buffer< Item, dim >::set_size( new_size );

                if (sizeOnDevice)
                {
                    cudaStream_t cuda_stream = 0;

                    CUPLA_KERNEL( KernelSetValueOnDeviceMemory )(
                        1,
                        1,
                        0,
                        cuda_stream
                    )(
                        this->getCurrentSizeOnDevicePointer(),
                        size
                    );
                }
            },
            TaskProperties::Builder()
               .label("DeviceBufferIntern::set_size()")
               .resources({
                   access_size_write(),
                   cuda_resources::streams[0].write()
               })
        );
    }

    void copyFrom(HostBuffer<TYPE, DIM>& other)
    {
        PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        memory::buffers::copy( *this, other );
    }

    void copyFrom(DeviceBuffer<TYPE, DIM>& other)
    {
        PMACC_ASSERT(this->isMyDataSpaceGreaterThan(other.getCurrentDataSpace()));
        memory::buffers::copy( *this, other );
    }

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

        auto guard = buffer::WriteGuard< DeviceBufferIntern<Item, Dim> >(
                         rg::SharedResourceObject(shared_from_this(), *this));

        if( isSmall )
            device_set_value_small( guard, value );
        else
            device_set_value_big( guard, value );
    }

private:

    /*! create native array with pitched lines
     */
    void createData()
    {
        Environment<>::task(
            [obj=shared_from_this()]
            {
                data.ptr = nullptr;
                data.pitch = 1;
                data.xsize = obj->getDataSpace()[0] * sizeof(Item);
                data.ysize = 1;

                if (DIM == DIM1)
                {
                    log<ggLog::MEMORY >("Create device 1D data: %1% MiB") % (data.xsize / 1024 / 1024);
                    CUDA_CHECK(cudaMallocPitch(&data.ptr, &data.pitch, data.xsize, 1));
                }
                if (DIM == DIM2)
                {
                    data.ysize = obj->getDataSpace()[1];
                    log<ggLog::MEMORY >("Create device 2D data: %1% MiB") % (data.xsize * data.ysize / 1024 / 1024);
                    CUDA_CHECK(cudaMallocPitch(&data.ptr, &data.pitch, data.xsize, data.ysize));
                }
                if (DIM == DIM3)
                {
                    cudaExtent extent;
                    extent.width = obj->getDataSpace()[0] * sizeof (TYPE);
                    extent.height = obj->getDataSpace()[1];
                    extent.depth = obj->getDataSpace()[2];

                    log<ggLog::MEMORY >("Create device 3D data: %1% MiB") % (obj->getDataSpace().productOfComponents() * sizeof(Item) / 1024 / 1024);
                    CUDA_CHECK(cudaMalloc3D(&data, extent));
                }

                reset(false);
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::createData()")
                .resources({
                    access_size_write( rg::access::Field<dim>(rg::access::IO::write) )
                    cuda_resources::streams[0].write()
                })
        );
    }

    /*!create 1D, 2D, 3D Array which use only a vector as base
     */
    void createFakeData()
    {
        Environment<>::task(
            [obj=shared_from_this()]
            {
                data.ptr = nullptr;
                data.pitch = 1;
                data.xsize = obj->getDataSpace()[0] * sizeof(Item);
                data.ysize = 1;

                log<ggLog::MEMORY >("Create device fake data: %1% MiB") % (obj->getDataSpace().productOfComponents() * sizeof(Item) / 1024 / 1024);
                CUDA_CHECK(cudaMallocPitch(&data.ptr, &data.pitch, obj->getDataSpace().productOfComponents() * sizeof(Item), 1));

                //fake the pitch, thus we can use this 1D Buffer as 2D or 3D
                data.pitch = obj->getDataSpace()[0] * sizeof(Item);

                if (DIM > DIM1)
                {
                    data.ysize = obj->getDataSpace()[1];
                }

                reset(false);
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::createFakeData()")
                .resources({
                    access_size_write( rg::access::Field<dim>(rg::access::IO::write) )
                    cuda_resources::streams[0].write()
                })
        );
    }

    void createSizeOnDevice(bool sizeOnDevice)
    {
        Environment<>::task(
            [obj=shared_from_this(), sizeOnDevice]
            {
                this->sizeOnDevicePtr = nullptr;

                if (sizeOnDevice)
                {
                    CUDA_CHECK(cudaMalloc((void**)&this->sizeOnDevicePtr, sizeof (size_t)));
                }

                setCurrentSize(this->getDataSpace().productOfComponents());
            },
            TaskProperties::Builder()
                .label("DeviceBufferIntern::createSizeOnDevice()")
                .resources({
                    access_size_write()
                    cuda_resources::streams[0].write()
                })
        );
    }

private:
    DataSpace<DIM> offset;

    bool sizeOnDevice;
    size_t* sizeOnDevicePtr;
    cuplaPitchedPtr data;
    bool useOtherMemory;
};

} //namespace pmacc
