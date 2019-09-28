
#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/mappings/threads/ForEachIdx.hpp"
#include "pmacc/mappings/threads/IdxConfig.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/nvidia/gpuEntryFunction.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits.hpp>

#include <pmacc/types.hpp>

namespace pmacc
{

template <class TYPE, std::size_t DIM>
class DeviceBuffer;

namespace memory
{
namespace buffers
{

namespace taskSetValueHelper
{

/** define access operation for non-pointer types
 */
template<typename T_Type, bool isPointer>
struct Value
{
    typedef const T_Type type;

    HDINLINE type& operator()(type& v) const
    {
        return v;
    }
};

/** define access operation for pointer types
 *
 * access first element of a pointer
 */
template<typename T_Type>
struct Value<T_Type, true>
{
    typedef const T_Type PtrType;
    typedef const typename boost::remove_pointer<PtrType>::type type;

    HDINLINE type& operator()(PtrType v) const
    {
        return *v;
    }
};

/** Get access to a value from a pointer or reference with the same method
 */
template<typename T_Type>
HDINLINE typename Value<T_Type, boost::is_pointer<T_Type>::value >::type&
getValue(T_Type& value)
{
    typedef Value<T_Type, boost::is_pointer<T_Type>::value > Functor;
    return Functor()(value);
}

}

/** set a value to all elements of a box
 *
 * @tparam T_numWorkers number of workers
 * @tparam T_xChunkSize number of elements in x direction to prepare with one cuda block
 */
template<
    uint32_t T_numWorkers,
    uint32_t T_xChunkSize
>
struct KernelSetValue
{
    /** set value to all elements
     *
     * @tparam T_DataBox pmacc::DataBox, type of the memory box
     * @tparam T_ValueType type of the value
     * @tparam T_SizeVecType pmacc::math::Vector, index type
     * @tparam T_Acc alpaka accelerator type
     *
     * @param memBox box of which all elements shall be set to value
     * @param value value to set to all elements of memBox
     * @param size extents of memBox
     */
    template<
        typename T_DataBox,
        typename T_ValueType,
        typename T_SizeVecType,
        typename T_Acc
    >
    DINLINE void
    operator()(
        T_Acc const & acc,
        T_DataBox & memBox,
        T_ValueType const & value,
        T_SizeVecType const & size
    ) const
    {
        using namespace mappings::threads;
        using SizeVecType = T_SizeVecType;

        SizeVecType const blockIndex( blockIdx );
        SizeVecType blockSize( SizeVecType::create( 1 ) );
        blockSize.x( ) = T_xChunkSize;

        constexpr uint32_t numWorkers = T_numWorkers;
        uint32_t const workerIdx = threadIdx.x;

        ForEachIdx<
            IdxConfig<
                T_xChunkSize,
                numWorkers
            >
        >{ workerIdx }(
            [&](
                uint32_t const linearIdx,
                uint32_t const
            )
            {
                auto virtualWorkerIdx( SizeVecType::create( 0 ) );
                virtualWorkerIdx.x( ) = linearIdx;

                SizeVecType const idx( blockSize * blockIndex + virtualWorkerIdx );
                if( idx.x() < size.x() )
                    memBox( idx ) = taskSetValueHelper::getValue( value );
            }
        );
    }
};

template <
    typename T,
    std::size_t T_Dim
>
void
device_set_value_small(DeviceBuffer<T, T_Dim> & dst, T const & value)
{
    Environment<>::get().ResourceManager().emplace_task(
        [&dst, value]
        {
            cudaStream_t cuda_stream = 0;

            // n-dimensional size of destination
            DataSpace< T_Dim > const area_size( dst.getCurrentDataSpace() );

            if( area_size.productOfComponents() != 0 )
            {
                auto gridSize = area_size;

                /* number of elements in x direction used to chunk the destination buffer
                 * for block parallel processing
                 */
                constexpr uint32_t xChunkSize = 256;
                constexpr uint32_t numWorkers = traits::GetNumWorkers<
                    xChunkSize
                    >::value;

                // number of blocks in x direction
                gridSize.x() = ceil(
                                    static_cast< double >( gridSize.x( ) ) /
                                    static_cast< double >( xChunkSize )
                                    );

                auto destBox = dst.getDataBox( );

                CUPLA_KERNEL(
                    KernelSetValue<
                        numWorkers,
                        xChunkSize
                    >
                )(
                    gridSize,
                    numWorkers,
                    0,
                    cuda_stream
                )(
                    destBox,
                    value,
                    area_size
                );
            }
        },
        TaskProperties::Builder()
           .label("device_set_value_small(" + std::to_string(value) + ")")
           .resources({
                dst.write(),
                dst.size_resource.write(),
                cuda_resources::streams[0].write()
            })
    );
}

/** implementation for big values (>256 byte)
 *
 * This class uses CUDA memcopy to copy an instance of T_ValueType to the GPU
 * and runs a kernel which assigns this value to all cells.
 */
template <
    typename T,
    std::size_t T_Dim
>
void
device_set_value_big(DeviceBuffer<T, T_Dim> & dst, T const & value)
{
    Environment<>::get().ResourceManager().emplace_task(
        [&dst, value]
        {
            cudaStream_t cuda_stream = 0;

            size_t current_size = dst.getCurrentSize();
            const DataSpace< T_Dim > area_size(dst.getCurrentDataSpace(current_size));
            if(area_size.productOfComponents() != 0)
            {
                auto gridSize = area_size;

                /* number of elements in x direction used to chunk the destination buffer
                 * for block parallel processing
                 */
                constexpr int xChunkSize = 256;
                constexpr uint32_t numWorkers = traits::GetNumWorkers<
                    xChunkSize
                    >::value;

                // number of blocks in x direction
                gridSize.x() = ceil(
                                    static_cast< double >( gridSize.x( ) ) /
                                    static_cast< double >( xChunkSize )
                                    );

                T* devicePtr = dst.getPointer();
                T* valuePointer_host;
                CUDA_CHECK( cudaMallocHost(
                                           (void**)&valuePointer_host,
                                           sizeof(T)
                                           ));
                *valuePointer_host = value; //copy value to new place

                CUDA_CHECK( cudaMemcpyAsync(
                                            devicePtr,
                                            valuePointer_host,
                                            sizeof(T),
                                            cudaMemcpyHostToDevice,
                                            cuda_stream
                                            ));

                auto destBox = dst.getDataBox( );
                CUPLA_KERNEL(
                    KernelSetValue<
                        numWorkers,
                        xChunkSize
                    >
                )(
                    gridSize,
                    numWorkers,
                    0,
                    cuda_stream
                )(
                    destBox,
                    devicePtr,
                    area_size
                );

                cudaDeviceSynchronize();

                if (valuePointer_host != nullptr)
                {
                    CUDA_CHECK_NO_EXCEPT(cudaFreeHost(valuePointer_host));
                    valuePointer_host = nullptr;
                }
            }
        },
        TaskProperties::Builder()
            .label("device_set_value_big(" + std::to_string(value) + ")")
            .resources({
                dst.write(),
                dst.size_resource.write(),
                cuda_resources::streams[0].write()
            })
    );
}

} // namespace buffers

} // namespace memory

} // namespace pmacc

