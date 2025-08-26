#!/usr/bin/env python3
"""
AWS Bedrock InvokeWithResponseStream Response Parser

This module parses binary response files from AWS Bedrock streaming responses
to Anthropic's Claude models. The responses use the AWS Event Stream format.
"""

import base64
import json
import struct
import binascii
from typing import List, Dict, Any, Iterator, Optional
import boto3


class BedrockStreamParser:
    """Parser for AWS Bedrock streaming response data."""
    
    def __init__(self):
        pass
    
    def parse_event_stream(self, data: bytes) -> Iterator[Dict[str, Any]]:
        """
        Parse AWS Event Stream format data.
        
        AWS Event Stream format consists of:
        - Prelude (12 bytes): total_length (4), headers_length (4), prelude_crc (4)
        - Headers: key-value pairs
        - Payload: the actual data
        - Message CRC (4 bytes)
        
        Args:
            data: Raw binary data from the response file
            
        Yields:
            Parsed event dictionaries
        """
        offset = 0
        
        while offset < len(data):
            try:
                # Parse prelude (12 bytes)
                if offset + 12 > len(data):
                    break
                    
                total_length, headers_length, prelude_crc = struct.unpack('>III', data[offset:offset+12])
                
                # Validate we have enough data for the complete message
                if offset + total_length > len(data):
                    break
                
                # Extract headers
                headers_start = offset + 12
                headers_data = data[headers_start:headers_start + headers_length]
                headers = self._parse_headers(headers_data)
                
                # Extract payload
                payload_start = headers_start + headers_length
                payload_length = total_length - 16 - headers_length  # total - prelude - headers - crc
                payload_data = data[payload_start:payload_start + payload_length]
                
                # Parse the event based on headers
                event = self._parse_event(headers, payload_data)
                if event:
                    yield event
                
                # Move to next message
                offset += total_length
                
            except (struct.error, ValueError, json.JSONDecodeError) as e:
                print(f"Error parsing at offset {offset}: {e}")
                break
    
    def _parse_headers(self, headers_data: bytes) -> Dict[str, str]:
        """Parse headers from the headers section."""
        headers = {}
        offset = 0
        
        while offset < len(headers_data):
            try:
                # Header name length (1 byte)
                name_length = headers_data[offset]
                offset += 1
                
                # Header name
                name = headers_data[offset:offset + name_length].decode('utf-8')
                offset += name_length
                
                # Header value type (1 byte) - we expect string type (7)
                value_type = headers_data[offset]
                offset += 1
                
                if value_type == 7:  # String type
                    # Value length (2 bytes)
                    value_length = struct.unpack('>H', headers_data[offset:offset+2])[0]
                    offset += 2
                    
                    # Value
                    value = headers_data[offset:offset + value_length].decode('utf-8')
                    offset += value_length
                    
                    headers[name] = value
                else:
                    # Skip unknown header types
                    break
                    
            except (struct.error, UnicodeDecodeError, IndexError):
                break
                
        return headers
    
    def _parse_event(self, headers: Dict[str, str], payload: bytes) -> Optional[Dict[str, Any]]:
        """Parse an individual event based on headers and payload."""
        event_type = headers.get(':event-type')
        message_type = headers.get(':message-type')
        
        if message_type == 'event' and event_type == 'chunk':
            try:
                # Parse the JSON payload
                payload_str = payload.decode('utf-8')
                chunk_data = json.loads(payload_str)
                
                # Decode base64 bytes if present
                if 'bytes' in chunk_data:
                    decoded_bytes = base64.b64decode(chunk_data['bytes'])
                    chunk_json = json.loads(decoded_bytes.decode('utf-8'))
                    
                    return {
                        'event_type': event_type,
                        'message_type': message_type,
                        'headers': headers,
                        'chunk_data': chunk_json
                    }
                    
            except (json.JSONDecodeError, UnicodeDecodeError, base64.binascii.Error):
                pass
                
        elif message_type == 'exception':
            # Handle exception events
            try:
                payload_str = payload.decode('utf-8')
                exception_data = json.loads(payload_str)
                return {
                    'event_type': 'exception',
                    'message_type': message_type,
                    'headers': headers,
                    'exception_data': exception_data
                }
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        return None
    
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Parse a Bedrock response file and return all events.
        
        Args:
            filepath: Path to the binary response file
            
        Returns:
            List of parsed events
        """
        with open(filepath, 'rb') as f:
            data = f.read()
        
        return list(self.parse_event_stream(data))
    
    def extract_text_content(self, events: List[Dict[str, Any]]) -> str:
        """
        Extract the complete text response from parsed events.
        
        Args:
            events: List of parsed events from parse_file()
            
        Returns:
            Complete text response
        """
        text_parts = []
        
        for event in events:
            if event.get('event_type') == 'chunk':
                chunk_data = event.get('chunk_data', {})
                
                # Handle content block delta events (text streaming)
                if chunk_data.get('type') == 'content_block_delta':
                    delta = chunk_data.get('delta', {})
                    if delta.get('type') == 'text_delta':
                        text_parts.append(delta.get('text', ''))
        
        return ''.join(text_parts)
    
    def get_message_info(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract message metadata from parsed events.
        
        Args:
            events: List of parsed events from parse_file()
            
        Returns:
            Dictionary with message metadata
        """
        info = {}
        
        for event in events:
            if event.get('event_type') == 'chunk':
                chunk_data = event.get('chunk_data', {})
                
                # Message start event
                if chunk_data.get('type') == 'message_start':
                    message = chunk_data.get('message', {})
                    info.update({
                        'message_id': message.get('id'),
                        'model': message.get('model'),
                        'role': message.get('role'),
                        'usage': message.get('usage', {})
                    })
                
                # Message delta event (final usage stats)
                elif chunk_data.get('type') == 'message_delta':
                    usage = chunk_data.get('usage', {})
                    if usage:
                        if 'usage' not in info:
                            info['usage'] = {}
                        info['usage'].update(usage)
                
                # Message stop event (final metrics)
                elif chunk_data.get('type') == 'message_stop':
                    metrics = chunk_data.get('amazon-bedrock-invocationMetrics', {})
                    if metrics:
                        info['invocation_metrics'] = metrics
        
        return info


class BedrockStreamUnparser:
    """Unparser to convert JSON events back to AWS Event Stream binary format."""
    
    def __init__(self):
        pass
    
    def _calculate_crc32(self, data: bytes) -> int:
        """Calculate CRC32 checksum for AWS Event Stream format."""
        return binascii.crc32(data) & 0xffffffff
    
    def _encode_headers(self, headers: Dict[str, str]) -> bytes:
        """
        Encode headers into AWS Event Stream format.
        
        Args:
            headers: Dictionary of header key-value pairs
            
        Returns:
            Binary encoded headers
        """
        header_data = b''
        
        for name, value in headers.items():
            # Header name length (1 byte)
            name_bytes = name.encode('utf-8')
            header_data += struct.pack('B', len(name_bytes))
            
            # Header name
            header_data += name_bytes
            
            # Header value type (1 byte) - 7 for string
            header_data += struct.pack('B', 7)
            
            # Header value length (2 bytes, big-endian)
            value_bytes = value.encode('utf-8')
            header_data += struct.pack('>H', len(value_bytes))
            
            # Header value
            header_data += value_bytes
        
        return header_data
    
    def _create_event_message(self, headers: Dict[str, str], payload: bytes) -> bytes:
        """
        Create a complete AWS Event Stream message.
        
        Args:
            headers: Dictionary of headers for the event
            payload: Binary payload data
            
        Returns:
            Complete binary message in AWS Event Stream format
        """
        # Encode headers
        headers_data = self._encode_headers(headers)
        
        # Calculate lengths
        headers_length = len(headers_data)
        payload_length = len(payload)
        total_length = 12 + headers_length + payload_length + 4  # prelude + headers + payload + crc
        
        # Create prelude (12 bytes)
        prelude = struct.pack('>III', total_length, headers_length, 0)  # CRC will be calculated
        prelude_crc = self._calculate_crc32(prelude[:8])
        prelude = struct.pack('>II', total_length, headers_length) + struct.pack('>I', prelude_crc)
        
        # Combine message parts (without final CRC)
        message_without_crc = prelude + headers_data + payload
        
        # Calculate final message CRC
        message_crc = self._calculate_crc32(message_without_crc)
        
        # Complete message
        complete_message = message_without_crc + struct.pack('>I', message_crc)
        
        return complete_message
    
    def create_chunk_event(self, chunk_data: Dict[str, Any]) -> bytes:
        """
        Create a chunk event in AWS Event Stream format.
        
        Args:
            chunk_data: The JSON data to be base64 encoded in the chunk
            
        Returns:
            Binary event stream message
        """
        # Encode the chunk data as JSON then base64
        json_str = json.dumps(chunk_data, separators=(',', ':'))
        encoded_bytes = base64.b64encode(json_str.encode('utf-8')).decode('ascii')
        
        # Create the payload structure
        payload_data = {
            "bytes": encoded_bytes,
            "p": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[:len(encoded_bytes)]
        }
        
        # Create headers
        headers = {
            ":event-type": "chunk",
            ":content-type": "application/json",
            ":message-type": "event"
        }
        
        # Encode payload as JSON
        payload = json.dumps(payload_data, separators=(',', ':')).encode('utf-8')
        
        return self._create_event_message(headers, payload)
    
    def create_message_start_event(self, message_id: str, model: str, usage: Dict[str, int]) -> bytes:
        """Create a message_start event."""
        chunk_data = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage
            }
        }
        return self.create_chunk_event(chunk_data)
    
    def create_content_block_start_event(self, index: int = 0) -> bytes:
        """Create a content_block_start event."""
        chunk_data = {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "text",
                "text": ""
            }
        }
        return self.create_chunk_event(chunk_data)
    
    def create_content_block_delta_event(self, text: str, index: int = 0) -> bytes:
        """Create a content_block_delta event."""
        chunk_data = {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "text_delta",
                "text": text
            }
        }
        return self.create_chunk_event(chunk_data)
    
    def create_content_block_stop_event(self, index: int = 0) -> bytes:
        """Create a content_block_stop event."""
        chunk_data = {
            "type": "content_block_stop",
            "index": index
        }
        return self.create_chunk_event(chunk_data)
    
    def create_message_delta_event(self, stop_reason: str = "end_turn", output_tokens: int = 0) -> bytes:
        """Create a message_delta event."""
        chunk_data = {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None
            },
            "usage": {
                "output_tokens": output_tokens
            }
        }
        return self.create_chunk_event(chunk_data)
    
    def create_message_stop_event(self, invocation_metrics: Dict[str, int]) -> bytes:
        """Create a message_stop event."""
        chunk_data = {
            "type": "message_stop",
            "amazon-bedrock-invocationMetrics": invocation_metrics
        }
        return self.create_chunk_event(chunk_data)
    
    def create_stream_from_text(self, text: str, message_id: str, model: str = "claude-3-5-haiku-20241022",
                               input_tokens: int = 10, chunk_size: int = 20) -> bytes:
        """
        Create a complete AWS Event Stream from text response.
        
        Args:
            text: The complete text response
            message_id: Unique message ID
            model: Model name
            input_tokens: Number of input tokens
            chunk_size: Average size of text chunks for streaming
            
        Returns:
            Complete binary AWS Event Stream
        """
        stream_data = b''
        
        # Calculate output tokens (rough estimate)
        output_tokens = len(text.split())
        
        # 1. Message start
        usage = {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 3  # Initial estimate
        }
        stream_data += self.create_message_start_event(message_id, model, usage)
        
        # 2. Content block start
        stream_data += self.create_content_block_start_event()
        
        # 3. Content block deltas (streaming text in chunks)
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            stream_data += self.create_content_block_delta_event(chunk_text)
        
        # 4. Content block stop
        stream_data += self.create_content_block_stop_event()
        
        # 5. Message delta
        stream_data += self.create_message_delta_event(output_tokens=output_tokens)
        
        # 6. Message stop
        invocation_metrics = {
            "inputTokenCount": input_tokens,
            "outputTokenCount": output_tokens,
            "invocationLatency": 1000,
            "firstByteLatency": 200
        }
        stream_data += self.create_message_stop_event(invocation_metrics)
        
        return stream_data
    
    def save_stream_to_file(self, stream_data: bytes, filepath: str):
        """Save binary stream data to a file."""
        with open(filepath, 'wb') as f:
            f.write(stream_data)


def main():
    """Example usage of the parser and unparser."""
    parser = BedrockStreamParser()
    unparser = BedrockStreamUnparser()
    
    # Test 1: Parse each response file
    print("=== PARSING ORIGINAL FILES ===")
    for i in range(1, 4):
        filename = f"resp{i}"
        print(f"\n--- Parsing {filename} ---")
        
        try:
            events = parser.parse_file(filename)
            text_content = parser.extract_text_content(events)
            message_info = parser.get_message_info(events)
            
            print(f"Number of events: {len(events)}")
            print(f"Text content: {repr(text_content)}")
            print(f"Message info: {json.dumps(message_info, indent=2)}")
            
        except FileNotFoundError:
            print(f"File {filename} not found")
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
    
    # Test 2: Create new stream from text and test round-trip
    print(f"\n\n=== TESTING UNPARSER (Round-trip conversion) ===")
    
    test_text = "Hello! This is a test message from the unparser."
    test_message_id = "msg_test_12345"
    
    # Create binary stream from text
    print("Creating binary stream from text...")
    binary_stream = unparser.create_stream_from_text(
        text=test_text,
        message_id=test_message_id,
        input_tokens=15,
        chunk_size=10
    )
    
    # Save to file
    test_filename = "test_generated.resp"
    unparser.save_stream_to_file(binary_stream, test_filename)
    print(f"Saved generated stream to {test_filename}")
    
    # Parse the generated file back
    print("Parsing the generated file...")
    try:
        generated_events = parser.parse_file(test_filename)
        generated_text = parser.extract_text_content(generated_events)
        generated_info = parser.get_message_info(generated_events)
        
        print(f"Generated events: {len(generated_events)}")
        print(f"Recovered text: {repr(generated_text)}")
        print(f"Generated info: {json.dumps(generated_info, indent=2)}")
        
        # Verify round-trip
        if generated_text == test_text:
            print("✅ Round-trip successful! Text matches perfectly.")
        else:
            print("❌ Round-trip failed! Text does not match.")
            print(f"Expected: {repr(test_text)}")
            print(f"Got: {repr(generated_text)}")
        
        if generated_info.get('message_id') == test_message_id:
            print("✅ Message ID matches!")
        else:
            print("❌ Message ID mismatch!")
            
    except Exception as e:
        print(f"Error testing round-trip: {e}")
    
    # Test 3: Demonstrate individual event creation
    print(f"\n\n=== TESTING INDIVIDUAL EVENT CREATION ===")
    
    print("Creating individual events...")
    
    # Create a simple message start event
    msg_start = unparser.create_message_start_event(
        message_id="msg_demo_67890",
        model="claude-3-5-haiku-20241022",
        usage={"input_tokens": 5, "cache_creation_input_tokens": 0, 
               "cache_read_input_tokens": 0, "output_tokens": 1}
    )
    
    # Create a text delta event
    text_delta = unparser.create_content_block_delta_event("Hello world!")
    
    # Save individual events to test file
    individual_test = "individual_events.resp"
    with open(individual_test, 'wb') as f:
        f.write(msg_start + text_delta)
    
    # Parse individual events
    try:
        individual_events = parser.parse_file(individual_test)
        print(f"Individual events parsed: {len(individual_events)}")
        for i, event in enumerate(individual_events):
            print(f"Event {i+1}: {event.get('chunk_data', {}).get('type', 'unknown')}")
            
    except Exception as e:
        print(f"Error parsing individual events: {e}")


if __name__ == "__main__":
    main()