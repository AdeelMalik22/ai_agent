#!/usr/bin/env python
"""Quick test script for file I/O tools."""

import os
import json
from pathlib import Path

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from tools import read_file, write_file, list_files

def test_file_tools():
    """Test the file I/O tools."""
    print("=" * 60)
    print("FILE I/O TOOLS TEST SUITE")
    print("=" * 60)

    # Set workspace root to current directory
    os.environ["WORKSPACE_ROOT"] = os.getcwd()

    # Test 1: List files in current directory
    print("\n[TEST 1] list_files('.')")
    print("-" * 60)
    result = list_files(".")
    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
    else:
        print(f"✅ SUCCESS: Found {result.get('count', 0)} entries")
        for entry in result.get("entries", [])[:5]:
            print(f"   - {entry['name']} ({entry['type']})")
        if result.get('count', 0) > 5:
            print(f"   ... and {result['count'] - 5} more")

    # Test 2: Read README.md
    print("\n[TEST 2] read_file('README.md')")
    print("-" * 60)
    result = read_file("documentation/README.md")
    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
    else:
        content = result.get("content", "")
        print(f"✅ SUCCESS: Read {result.get('size_bytes', 0)} bytes")
        print(f"   First 100 chars: {content[:100]}...")

    # Test 3: Write a test file
    print("\n[TEST 3] write_file('test_output.txt')")
    print("-" * 60)
    test_content = "This is a test file created by the file tools test script.\n"
    result = write_file("test_output.txt", test_content)
    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
    else:
        print(f"✅ SUCCESS: {result.get('message', '')}")

    # Test 4: Read the file we just wrote
    print("\n[TEST 4] read_file('test_output.txt')")
    print("-" * 60)
    result = read_file("test_output.txt")
    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
    else:
        content = result.get("content", "")
        print(f"✅ SUCCESS: Read {result.get('size_bytes', 0)} bytes")
        print(f"   Content: {content.strip()}")

    # Test 5: List config directory
    print("\n[TEST 5] list_files('config')")
    print("-" * 60)
    result = list_files("config")
    if "error" in result:
        print(f"❌ ERROR: {result['error']}")
    else:
        print(f"✅ SUCCESS: Found {result.get('count', 0)} entries in config/")
        for entry in result.get("entries", []):
            print(f"   - {entry['name']} ({entry['type']})")

    # Test 6: Try to read a non-existent file
    print("\n[TEST 6] read_file('nonexistent.txt') - Expected to fail")
    print("-" * 60)
    result = read_file("nonexistent.txt")
    if "error" in result:
        print(f"✅ EXPECTED ERROR: {result['error']}")
    else:
        print(f"❌ UNEXPECTED: File should not exist")

    # Test 7: Try path traversal (should be blocked)
    print("\n[TEST 7] read_file('../etc/passwd') - Should be blocked")
    print("-" * 60)
    result = read_file("../etc/passwd")
    if "error" in result:
        print(f"✅ BLOCKED (as expected): {result['error']}")
    else:
        print(f"❌ SECURITY ISSUE: Path traversal was not blocked!")

    # Cleanup
    print("\n[CLEANUP] Removing test_output.txt")
    print("-" * 60)
    try:
        Path("test_output.txt").unlink()
        print("✅ Test file cleaned up")
    except Exception as e:
        print(f"⚠️ Could not clean up: {e}")

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_file_tools()

