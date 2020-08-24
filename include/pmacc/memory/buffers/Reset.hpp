
#pragma once

#include <pmacc/Environment.hpp>

#include <pmacc/memory/buffers/hostBuffer/Fill.hpp>
#include <pmacc/memory/buffers/deviceBuffer/Fill.hpp>
#include <pmacc/memory/buffers/Buffer.hpp>

namespace pmacc
{
namespace mem
{
namespace buffer
{

/*!
 * Reset the buffer size to its data space and
 * clear its content if preserve_data is false.
 * Requires that fill() is implemented for the buffer type
 *
 * todo: should this always clear?, the size can
 *       be reset with  buffer.size().reset()
 *
 */
template< typename Buffer >
auto reset(
    WriteGuard< Buffer > buffer,
    bool preserve_data = true
)
{
    return Environment<>::task(
        [preserve_data]( auto buffer )
        {
            buffer.size().reset();

            if( !preserve_data )
            {
                using Item = typename Buffer::Item;
                Item value;

                /* using `uint8_t` for byte-wise looping through tmp var value of `Item` */
                uint8_t * valuePtr = (uint8_t *) &value;
                for( size_t b = 0; b < sizeof(Item); ++b)
                    valuePtr[b] = static_cast<uint8_t>(0);

                /* set value with zero-ed `TYPE` */
                pmacc::mem::buffer::fill( buffer, value );
            }
        },
        TaskProperties::Builder()
            .label("reset Buffer"),
        buffer.write()
    );
}

} // namespace buffer

} // namespace mem

} // namespace pmacc

