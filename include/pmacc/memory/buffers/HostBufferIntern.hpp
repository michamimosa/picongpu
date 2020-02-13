/* Copyright 2013-2020 Rene Widera, Benjamin Worpitz,
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

#include "pmacc/memory/buffers/HostBuffer.hpp"
#include "pmacc/memory/boxes/DataBoxDim1Access.hpp"
#include "pmacc/assert.hpp"

//#include <pmacc/memory/buffers/CopyDeviceToHost.hpp>
#include <pmacc/Environment.hpp>

namespace pmacc
{
namespace mem
{

/**
 * Internal implementation of the HostBuffer interface.
 */
template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
class HostBufferIntern
    : public HostBuffer< T_Item, T_dim, T_DataAccessPolicy >
{
private:
    std::shared_ptr< HostBufferIntern< T_Item, T_dim, T_DataAccessPolicy > > shared_from_this()
    {
        return this->template shared_from_base< HostBufferIntern<T_Item, T_dim, T_DataAccessPolicy> >();
    }

public:
    using Item = T_Item;
    static constexpr size_t dim = T_dim;
    typedef typename HostBuffer<Item, dim, T_DataAccessPolicy>::DataBoxType DataBoxType;

    /** constructor
     *
     * @param size extent for each dimension (in elements)
     */
    void init(DataSpace< dim > size)
    {
        this->HostBuffer<Item, dim, T_DataAccessPolicy>::init( size, size );
        Environment<>::task(
            [size]( auto x )
            {
                auto obj = std::static_pointer_cast< HostBufferIntern >( x );
                obj->ownPointer = true;
                CUDA_CHECK(cudaMallocHost((void**)&obj->pointer, size.productOfComponents() * sizeof(Item)));
                obj->reset( false );
            },
            TaskProperties::Builder()
                .label("HostBufferIntern::HostBufferIntern()"),
            this->buffer_resource().write()
        );
    }

    void init(BufferResource< HostBufferIntern > source,
              DataSpace< dim > size,
              DataSpace< dim > offset = DataSpace<dim>())
    {
        this->HostBuffer<Item, dim, T_DataAccessPolicy>::init( size, source->getPhysicalMemorySize(), source );
        Environment<>::task(
            [obj=this->shared_from_this(), offset]( auto source )
            {
                obj->ownPointer = false;
                obj->pointer = &(source.getDataBox()(offset));/*fix me, this is a bad way*/
            },
            TaskProperties::Builder()
                .label("HostBufferIntern::HostBufferIntern(source)")
                .resources({ this->write_data() }),
            source
        );
    }

    /**
     * destructor
     */
    virtual ~HostBufferIntern()
    {
        CUDA_CHECK_NO_EXCEPT(cudaFreeHost(this->pointer));
    }

    /*! Get pointer of memory
     * @return pointer to memory
     */
    Item * getBasePointer() const
    {
        return pointer;
    }

    Item * getPointer() const
    {
        return pointer;
    }

    void reset(bool preserveData = true)
    {
        Environment<>::task(
            [preserveData]( auto x )
            {
                auto obj = std::static_pointer_cast< HostBufferIntern >( x );

                obj->set_size(obj->getDataSpace().productOfComponents());
                if (!preserveData)
                {
                    /* if it is a pointer out of other memory we can not assume that
                     * that the physical memory is contiguous
                     */
                    if(obj->ownPointer)
                        memset(obj->pointer, 0, obj->getDataSpace().productOfComponents() * sizeof(Item));
                    else
                    {
                        Item value;
                        /* using `uint8_t` for byte-wise looping through tmp var value of `Item` */
                        uint8_t * valuePtr = (uint8_t *) &value;
                        for( size_t b = 0; b < sizeof(Item); ++b)
                        {
                            valuePtr[b] = static_cast<uint8_t>(0);
                        }
                        /* set value with zero-ed `TYPE` */
                        obj->fill( value );
                    }
                }
            },
            TaskProperties::Builder()
                .label("HostBufferIntern::reset()"),
            this->buffer_resource().write()
        );
    }

    void fill(Item const & value)
    {
        Environment<>::task(
            [value]( auto x )
            {
                auto obj = std::static_pointer_cast< HostBufferIntern >( x );

                int64_t current_size = static_cast< int64_t >(obj->get_size());
                DataBoxDim1Access< DataBoxType > d1Box(
                    obj->getDataBox(),
                    obj->getDataSpace()
                );

                #pragma omp parallel for
                for (int64_t i = 0; i < current_size; i++)
                {
                    d1Box[i] = value;
                }
            },
            TaskProperties::Builder()
                .label("HostBufferIntern::setValue(" + std::to_string(value) + ")"),
            this->buffer_resource().write()
        );
    }

    DataBoxType getDataBox()
    {
        return DataBoxType(
                   PitchedBox<Item, dim>(
                       pointer,
                       DataSpace<dim>(),
                       this->getPhysicalMemorySize(),
                       this->getPhysicalMemorySize()[0] * sizeof (Item)
                   )
               );
    }

private:
    Item * pointer;
    bool ownPointer;
};

} // namespace mem

} // namespace pmacc

