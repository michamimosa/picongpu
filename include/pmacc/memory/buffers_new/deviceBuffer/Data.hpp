
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>

#include <pmacc/memory/buffers_new/common/Data.hpp>

#pragma once

namespace pmacc
{
namespace mem
{
namespace device_buffer
{

template <
    typename T_Item,
    std::size_t T_dim
>
struct DeviceBufferData
    : buffer::BufferData< T_Item, T_dim >
{
public:
    using Item = T_Item;
    static constexpr std::size_t dim = T_dim;
    using DataBoxType = DataBox< PitchedBox< Item, dim > >;

    DeviceBufferData(
        DataSpace< dim > capacity,
        bool use_vector_as_base
    )
        : capacity( capacity )
    {
        if( use_vector_as_base )
            createFakeData();
        else
            createData();
    }

    ~DeviceBufferData()
    {
        CUDA_CHECK_NO_EXCEPT(cudaFree(pitched_ptr.ptr));
    }

    DataSpace< dim > get_capacity() const noexcept
    {
        return capacity;
    }

    cudaPitchedPtr get_cuda_pitched() const noexcept
    {
        return pitched_ptr;
    }

    size_t get_pitch() const noexcept
    {
        return pitched_ptr.pitch;
    }
    size_t getPitch() const noexcept
    {
        return pitched_ptr.pitch;
    }

    Item * get_base_ptr() const noexcept
    {
        return (Item *)pitched_ptr.ptr;
    }

    Item * get_pointer( DataSpace< dim > offset ) const noexcept
    {
        if ( dim == DIM1 )
            return (Item*) get_base_ptr() + offset[0];

        else if ( dim == DIM2 )
            return (Item*) ((char*) get_base_ptr() + offset[1] * get_pitch()) + offset[0];

        else
        {
            //! TODO: could be more readable
            const size_t offsetY = offset[1] * get_pitch();
            const size_t sizePlaneXY = this->capacity[1] * get_pitch();
            return (Item*) ((char*) pitched_ptr.ptr + offset[2] * sizePlaneXY + offsetY) + offset[0];
        }
    }

    DataBoxType get_data_box( DataSpace< dim > offset ) const noexcept
    {
        return DataBoxType(
                   PitchedBox< Item, dim >(
                       get_base_ptr(),
                       offset,
                       get_capacity(),
                       get_pitch()
                   )
               );
    }

private:
    void createData()
    {
        pitched_ptr.ptr = nullptr;
        pitched_ptr.pitch = 1;
        pitched_ptr.xsize = capacity[0] * sizeof(Item);
        pitched_ptr.ysize = 1;

        if (dim == DIM1)
        {
            log<ggLog::MEMORY >("Create device 1D data: %1% MiB") % (pitched_ptr.xsize / 1024 / 1024);
            CUDA_CHECK(cudaMallocPitch(&pitched_ptr.ptr, &pitched_ptr.pitch, pitched_ptr.xsize, 1));
        }
        if (dim == DIM2)
        {
            pitched_ptr.ysize = capacity[1];
            log<ggLog::MEMORY >("Create device 2D data: %1% MiB") % (pitched_ptr.xsize * pitched_ptr.ysize / 1024 / 1024);
            CUDA_CHECK(cudaMallocPitch(
                &pitched_ptr.ptr,
                &pitched_ptr.pitch,
                pitched_ptr.xsize,
                pitched_ptr.ysize));
        }
        if (dim == DIM3)
        {
            cudaExtent extent;
            extent.width = capacity[0] * sizeof (Item);
            extent.height = capacity[1];
            extent.depth = capacity[2];

            log<ggLog::MEMORY >("Create device 3D data: %1% MiB") % (capacity.productOfComponents() * sizeof(Item) / 1024 / 1024);
            CUDA_CHECK(cudaMalloc3D(&pitched_ptr, extent));
        }
    }

    void createFakeData()
    {
        pitched_ptr.ptr = nullptr;
        pitched_ptr.pitch = 1;
        pitched_ptr.xsize = capacity[0] * sizeof(Item);
        pitched_ptr.ysize = 1;

        log< ggLog::MEMORY >("Create device fake data: %1% MiB") % (capacity.productOfComponents() * sizeof(Item) / 1024 / 1024);
        CUDA_CHECK(cudaMallocPitch(&pitched_ptr.ptr, &pitched_ptr.pitch, capacity.productOfComponents() * sizeof(Item), 1));

        //fake the pitch, thus we can use this 1D Buffer as 2D or 3D
        pitched_ptr.pitch = capacity[0] * sizeof(Item);

        if (dim > DIM1)
            pitched_ptr.ysize = capacity[1];
    }

    cudaPitchedPtr pitched_ptr;
    DataSpace< dim > capacity;
};

} // namespace device_buffer

} // namespace mem

} // namespace pmacc

