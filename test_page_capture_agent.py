#!/usr/bin/env python3
"""Test script for PageCaptureAgent with Computer Use."""

import asyncio
import os
from src.cv_writer_mcp.page_capture_agent import PageCaptureAgent
from src.cv_writer_mcp.models import PageCaptureRequest

async def test_page_capture_agent():
    """Test the PageCaptureAgent with Computer Use tool."""
    
    # Set OpenAI API key (would normally be in environment)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No OPENAI_API_KEY set, using placeholder")
        api_key = "test-key"
    
    try:
        # Create agent
        agent = PageCaptureAgent(api_key=api_key)
        
        # Create request
        request = PageCaptureRequest(pdf_file_path="output/to_improve_style.pdf")
        
        print(f"Testing PageCaptureAgent with Computer Use tool")
        print(f"PDF file: {request.pdf_file_path}")
        
        # Run the analysis
        response = await agent.capture_and_analyze(request)
        
        print(f"\nResults:")
        print(f"Status: {response.status}")
        print(f"Pages analyzed: {response.pages_analyzed}")
        print(f"Message: {response.message}")
        
        if response.visual_issues:
            print(f"\nVisual issues found: {len(response.visual_issues)}")
            for i, issue in enumerate(response.visual_issues[:3], 1):  # Show first 3
                print(f"  {i}. {issue}")
        
        if response.suggested_fixes:
            print(f"\nSuggested fixes: {len(response.suggested_fixes)}")
            for i, fix in enumerate(response.suggested_fixes[:3], 1):  # Show first 3
                print(f"  {i}. {fix}")
        
        if response.analysis_summary:
            print(f"\nAnalysis summary: {response.analysis_summary}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_page_capture_agent())