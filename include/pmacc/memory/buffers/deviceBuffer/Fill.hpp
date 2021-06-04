
#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include <pmacc/memory/Array.hpp>
#include <pmacc/lockstep.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits.hpp>

#include <pmacc/types.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/memory/buffers/deviceBuffer/Resource.hpp>

namespace pmacc
{
    namespace mem
    {
        namespace buffer
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
                HDINLINE typename Value<T_Type, boost::is_pointer<T_Type>::value>::type& getValue(T_Type& value)
                {
                    typedef Value<T_Type, boost::is_pointer<T_Type>::value> Functor;
                    return Functor()(value);
                }

            } // namespace taskSetValueHelper

            /** set a value to all elements of a box
             *
             * @tparam T_numWorkers number of workers
             * @tparam T_xChunkSize number of elements in x direction to prepare with one cuda block
             */
            template<uint32_t T_numWorkers, uint32_t T_xChunkSize>
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
                template<typename T_DataBox, typename T_ValueType, typename T_SizeVecType, typename T_Acc>
                DINLINE void operator()(
                    T_Acc const& acc,
                    T_DataBox& memBox,
                    T_ValueType const& value,
                    T_SizeVecType const& size) const
                {
                    using SizeVecType = T_SizeVecType;

                    SizeVecType const blockIndex(cupla::blockIdx(acc));
                    SizeVecType blockSize(SizeVecType::create(1));
                    blockSize.x() = T_xChunkSize;

                    constexpr uint32_t numWorkers = T_numWorkers;
                    uint32_t const workerIdx = cupla::threadIdx(acc).x;

                    lockstep::makeForEach<T_xChunkSize, numWorkers>(workerIdx)([&](uint32_t const linearIdx) {
                        auto virtualWorkerIdx(SizeVecType::create(0));
                        virtualWorkerIdx.x() = linearIdx;

                        SizeVecType const idx(blockSize * blockIndex + virtualWorkerIdx);
                        if(idx.x() < size.x())
                            memBox(idx) = taskSetValueHelper::getValue(value);
                    });
                }
            };

            template<typename T_Item, std::size_t T_Dim, typename T_DataAccessPolicy>
            void device_set_value_small(
                device_buffer::WriteGuard<T_Item, T_Dim, T_DataAccessPolicy> const& dst,
                T_Item && value)
            {
                Environment<>::task(
                    [value{std::move(value)}](auto dst) {
                        // n-dimensional size of destination
                        auto const area_size = dst.size().getCurrentDataSpace();

                        if(area_size.productOfComponents() != 0)
                        {
                            auto gridSize = area_size;

                            /* number of elements in x direction used to chunk the destination buffer
                             * for block parallel processing
                             */
                            constexpr uint32_t xChunkSize = 256;
                            constexpr uint32_t numWorkers = traits::GetNumWorkers<xChunkSize>::value;

                            // number of blocks in x direction
                            gridSize.x() = ceil(static_cast<double>(gridSize.x()) / static_cast<double>(xChunkSize));

                            Environment<>::task(
                                [numWorkers, xChunkSize, gridSize, area_size, value{std::move(value)}](auto dst) {
                                    CUPLA_KERNEL(KernelSetValue<numWorkers, xChunkSize>)
                                    (gridSize,
                                     numWorkers,
                                     0,
                                     redGrapes::thread::current_cupla_stream)(dst.getDataBox(), value, area_size);
                                },
                                TaskProperties::Builder()
                                    .label("DeviceSetValueSmall: cupla kernel")
                                    .scheduling_tags({SCHED_CUPLA}),
                                dst.data());
                        }
                    },
                    TaskProperties::Builder().label("device_set_value_small()" /* + std::to_string(value) + ")"*/),
                    dst.write());
            }

            /** implementation for big values (>256 byte)
             *
             * This class uses CUDA memcopy to copy an instance of T_ValueType to the GPU
             * and runs a kernel which assigns this value to all cells.
             */
            template<typename T_Item, std::size_t T_Dim, typename T_DataAccessPolicy>
            void device_set_value_big(
                device_buffer::WriteGuard<T_Item, T_Dim, T_DataAccessPolicy> const& dst,
                T_Item && value)
            {
                Environment<>::task(
                    [value{std::move(value)}](auto dst) {
                        auto const area_size = dst.size().getCurrentDataSpace();
                        if(area_size.productOfComponents() != 0)
                        {
                            auto gridSize = area_size;

                            /* number of elements in x direction used to chunk the destination buffer
                             * for block parallel processing
                             */
                            constexpr int xChunkSize = 256;
                            constexpr uint32_t numWorkers = traits::GetNumWorkers<xChunkSize>::value;

                            // number of blocks in x direction
                            gridSize.x() = ceil(static_cast<double>(gridSize.x()) / static_cast<double>(xChunkSize));

                            rg::IOResource<T_Item*> valuePointer_host;

                            Environment<>::task(
                                [value{std::move(value)}](auto valuePointer_host) {
                                    T_Item* ptr;
                                    CUDA_CHECK(cuplaMallocHost((void**) &ptr, sizeof(T_Item)));

                                    *ptr = value; // copy value to new place
                                    *valuePointer_host = ptr;
                                },
                                valuePointer_host.write());

                            Environment<>::task(
                                [gridSize, area_size](auto data, auto valuePointer_host) {
                                    CUDA_CHECK(cuplaMemcpyAsync(
                                        data.getPointer(),
                                        *valuePointer_host,
                                        sizeof(T_Item),
                                        cuplaMemcpyHostToDevice,
                                        redGrapes::thread::current_cupla_stream));

                                    CUPLA_KERNEL(KernelSetValue<numWorkers, xChunkSize>)
                                    (gridSize, numWorkers, 0, redGrapes::thread::current_cupla_stream)(
                                        data.getDataBox(),
                                        data.getPointer(),
                                        area_size);
                                },
                                TaskProperties::Builder().label("KernelSetValue").scheduling_tags({SCHED_CUPLA}),

                                dst.data().write(),
                                valuePointer_host.read());

                            Environment<>::task(
                                [](auto valuePointer_host) {
                                    CUDA_CHECK_NO_EXCEPT(cuplaFreeHost(*valuePointer_host));
                                },
                                valuePointer_host.write());
                        }
                    },
                    TaskProperties::Builder().label("device_set_value_big()" /* + std::to_string(value) + ")"*/),
                    dst.write());
            }


            template<typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy, bool T_is_small>
            struct FillHelper
            {
                auto operator()(
                    device_buffer::WriteGuard<T_Item, T_dim, T_DataAccessPolicy> const& device_buffer,
                    T_Item && value)
                {
                    return device_set_value_big(device_buffer, std::forward<T_Item>(value));
                }
            };

            template<typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy>
            struct FillHelper<T_Item, T_dim, T_DataAccessPolicy, true>
            {
                auto operator()(
                    device_buffer::WriteGuard<T_Item, T_dim, T_DataAccessPolicy> const& device_buffer,
                    T_Item && value)
                {
                    return device_set_value_small(device_buffer, std::forward<T_Item>(value));
                }
            };

            template<typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy>
            auto fill(device_buffer::WriteGuard<T_Item, T_dim, T_DataAccessPolicy> const& device_buffer, T_Item && value)
            {
                enum
                {
                    isSmall = (sizeof(T_Item) <= 128)
                }; // if we use const variable the compiler create warnings

                FillHelper<T_Item, T_dim, T_DataAccessPolicy, isSmall>{}(device_buffer, std::forward<T_Item>(value));
            }

            template<typename T_Item, std::size_t T_array_size, std::size_t T_dim, typename T_DataAccessPolicy>
            auto fill(
                device_buffer::WriteGuard<pmacc::memory::Array<T_Item, T_array_size>, T_dim, T_DataAccessPolicy> const&
                    device_buffer,
                T_Item && value)
            {
                fill(device_buffer, pmacc::memory::Array<T_Item, T_array_size>(std::forward<T_Item>(value)));
            }


        } // namespace buffer

    } // namespace mem

} // namespace pmacc
