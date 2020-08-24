
#pragma once

#include <redGrapes/resource/ioresource.hpp>
#include <pmacc/Environment.hpp>

namespace rg = redGrapes;

namespace pmacc
{
namespace mem
{
namespace buffer
{

template < std::size_t T_dim >
struct BufferSize
{
public:
    static constexpr std::size_t dim = T_dim;

    BufferSize()
    {
        size_t * host_size_ptr;
        CUDA_CHECK(cuplaMallocHost( (void**)&host_size_ptr, sizeof(size_t) ));
        this->host_current_size =
            rg::IOResource< size_t >(
                std::shared_ptr< size_t >(
                    host_size_ptr,
                    []( size_t * ptr )
                    {
                        CUDA_CHECK_NO_EXCEPT(cuplaFreeHost(ptr));
                    }));
    }

    virtual ~BufferSize() {}

    void reset() const
    {
        set_current_size( data_space.productOfComponents() );
    }

    void reset( DataSpace< dim > data_space )
    {
        this->data_space = data_space;
        reset();
    }

    /*! Get current size of any dimension
     * @return count of current elements per dimension
     */
    virtual size_t get_current_size() const
    {
        return Environment<>::task(
            []( auto host_current_size )
            {
                return *host_current_size;
            },
            TaskProperties::Builder()
                .label("BufferSize::get_current_size()"),
            host_current_size.read()
        ).get();
    }

    virtual void set_current_size( size_t new_size ) const
    {
        Environment<>::task(
            [new_size]( auto host_current_size )
            {
                *host_current_size = new_size;
            },
            TaskProperties::Builder()
                .label("BufferSize::set_current_size()"),
            host_current_size.write()
        );
    }

    /*! Get max spread (elements) of any dimension
     * @return spread (elements) per dimension
     */
    DataSpace< dim > get_data_space() const noexcept
    {
        return data_space;
    }

    /*! Spread of memory per dimension which is currently used
     * @return if DIM == DIM1 than return count of elements (x-direction)
     * if DIM == DIM2 than return how many lines (y-direction) of memory is used
     * if DIM == DIM3 than return how many slides (z-direction) of memory is used3
     */
    DataSpace< dim > get_current_data_space() const
    {
        DataSpace<dim> tmp;
        int64_t current_size = get_current_size();

        //!\todo: current size can be changed if it is a DeviceBuffer and current size is on device
        //call first get current size (but const not allow this)

        if (dim == DIM1)
        {
            tmp[0] = current_size;
        }
        if (dim == DIM2)
        {
            if (current_size <= data_space[0])
            {
                tmp[0] = current_size;
                tmp[1] = 1;
            } else
            {
                tmp[0] = data_space[0];
                tmp[1] = (current_size+data_space[0]-1) / data_space[0];
            }
        }
        if (dim == DIM3)
        {
            if (current_size <= data_space[0])
            {
                tmp[0] = current_size;
                tmp[1] = 1;
                tmp[2] = 1;
            } else if (current_size <= (data_space[0] * data_space[1]))
            {
                tmp[0] = data_space[0];
                tmp[1] = (current_size+data_space[0]-1) / data_space[0];
                tmp[2] = 1;
            } else
            {
                tmp[0] = data_space[0];
                tmp[1] = data_space[1];
                tmp[2] = (current_size+(data_space[0] * data_space[1])-1) / (data_space[0] * data_space[1]);
            }
        }

        return tmp;
    }

protected:
    
    template < typename, typename >
    friend class rg::trait::BuildProperties;

    DataSpace< dim > data_space;
    rg::IOResource< size_t > host_current_size;
};

} // namespace buffer

} // namespace mem

} // namespace pmacc

