
#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/mappings/threads/ForEachIdx.hpp"
#include "pmacc/mappings/threads/IdxConfig.hpp"
#include "pmacc/memory/buffers/DeviceBuffer.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/eventSystem/EventSystem.hpp"
#include "pmacc/eventSystem/tasks/StreamTask.hpp"
#include "pmacc/nvidia/gpuEntryFunction.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits.hpp>

#include <pmacc/types.hpp>
#include <pmacc/tasks/StreamTask.hpp>
#include <rmngr/task.hpp>

namespace pmacc
{

  template <class TYPE, unsigned DIM>
  class DeviceBuffer;

namespace NEW{
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

/** Set all cells of a GridBuffer on the device to a given value
 *
 * T_ValueType  = data type (e.g. float, float2)
 * T_dim   = dimension of the GridBuffer
 * T_isSmallValue = true if T_ValueType can be send via kernel parameter (on cuda T_ValueType must be smaller than 256 byte)
 */
template <class T_ValueType, size_t T_dim, bool T_isSmallValue>
class TaskSetValue;

template < typename T, size_t T_Dim >
struct SetValueTask
{
    void properties( Scheduler::SchedulablePtr s )
    {
        auto & l = s->proto_property< rmngr::ResourceUserPolicy >().access_list;
        l.push_back( this->destination->write() );
        l.push_back( this->destination->size_resource.write() );

        s->proto_property< GraphvizPolicy >().label = "SetValueOnDevice";
    }

    DeviceBuffer<T, T_Dim> * destination;
};

template <
    typename Impl,
    typename T,
    size_t T_Dim
>
class TaskSetValueBase
    : public rmngr::Task<
          Impl,
          boost::mpl::vector<
              StreamTask,
              SetValueTask< T, T_Dim >
          >
      >
{
public:
    TaskSetValueBase(DeviceBuffer<T, T_Dim> & dst, T const & value)
      : value(value)
    {
        this->destination = &dst;
    }

    virtual void run() = 0;

protected:
    T value;
};

/** implementation for small values (<= 256byte)
 */
template <typename T_ValueType, size_t T_Dim>
class TaskSetValue<T_ValueType, T_Dim, true>
    : public TaskSetValueBase<
          TaskSetValue<T_ValueType, T_Dim, true>,
          T_ValueType,
          T_Dim
      >
{
public:
    typedef T_ValueType ValueType;
    static constexpr uint32_t dim = T_Dim;

    TaskSetValue(DeviceBuffer<ValueType, dim>& dst, const ValueType& value)
      : TaskSetValueBase<TaskSetValue<T_ValueType, T_Dim, true>, ValueType, dim>(dst, value)
    {}

    void run()
    {
        // n-dimensional size of destination
        DataSpace< dim > const area_size( this->destination->getCurrentDataSpace() );

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

            auto destBox = this->destination->getDataBox( );
            CUPLA_KERNEL(
                KernelSetValue<
                    numWorkers,
                    xChunkSize
                >
            )(
                gridSize,
                numWorkers,
                0,
                this->getCudaStream( )
            )(
                destBox,
                this->value,
                area_size
            );
        }
    }
};

/** implementation for big values (>256 byte)
 *
 * This class uses CUDA memcopy to copy an instance of T_ValueType to the GPU
 * and runs a kernel which assigns this value to all cells.
 */
template <class T_ValueType, size_t T_Dim>
class TaskSetValue<T_ValueType, T_Dim, false>
    : public TaskSetValueBase<
          TaskSetValue<T_ValueType, T_Dim, false>,
          T_ValueType,
          T_Dim
      >
{
public:
    typedef T_ValueType ValueType;
    static constexpr uint32_t dim = T_Dim;

    TaskSetValue(DeviceBuffer<ValueType, dim>& dst, const ValueType& value)
        : TaskSetValueBase<TaskSetValue<T_ValueType, T_Dim, false>, ValueType, T_Dim>(dst, value)
        , valuePointer_host(nullptr)
    {}

    void run()
    {
        size_t current_size = this->destination->getCurrentSize();
        const DataSpace<dim> area_size(this->destination->getCurrentDataSpace(current_size));
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

            ValueType* devicePtr = this->destination->getPointer();

            CUDA_CHECK( cudaMallocHost(
                (void**)&valuePointer_host,
                sizeof( ValueType )
            ));
            *valuePointer_host = this->value; //copy value to new place

            CUDA_CHECK( cudaMemcpyAsync(
                devicePtr,
                valuePointer_host,
                sizeof( ValueType ),
                cudaMemcpyHostToDevice,
                this->getCudaStream( )
            ));

            auto destBox = this->destination->getDataBox( );
            CUPLA_KERNEL(
                KernelSetValue<
                    numWorkers,
                    xChunkSize
                >
            )(
                gridSize,
                numWorkers,
                0,
                this->getCudaStream()
            )(
                destBox,
                devicePtr,
                area_size
            );

            if (valuePointer_host != nullptr)
            {
                CUDA_CHECK_NO_EXCEPT(cudaFreeHost(valuePointer_host));
                valuePointer_host = nullptr;
            }
        }
    }

private:
    ValueType *valuePointer_host;
};

}
} //namespace pmacc

