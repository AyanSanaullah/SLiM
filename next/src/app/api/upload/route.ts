import { NextRequest, NextResponse } from 'next/server';

// Backend URL - you can set this in environment variables
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const chatId = formData.get('chatId') as string;

    // Validate file
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Check file size (limit to 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      return NextResponse.json(
        { error: 'File too large. Maximum size is 10MB.' },
        { status: 400 }
      );
    }

    // Create form data for backend
    const backendFormData = new FormData();
    backendFormData.append('file', file);
    if (chatId) backendFormData.append('chatId', chatId);

    // Forward the request to your Python backend
    const backendResponse = await fetch(`${BACKEND_URL}/api/upload`, {
      method: 'POST',
      body: backendFormData,
    });

    // Check if backend request was successful
    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error('Backend upload error:', errorText);
      return NextResponse.json(
        { error: 'File upload failed' },
        { status: 502 }
      );
    }

    // Parse backend response
    const data = await backendResponse.json();

    // Return the response from backend
    return NextResponse.json({
      success: true,
      fileUrl: data.fileUrl || data.url,
      fileName: file.name,
      fileSize: file.size,
      fileType: file.type,
      chatId: data.chatId || chatId,
      message: data.message || 'File uploaded successfully',
    });

  } catch (error) {
    console.error('Upload API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error during file upload' },
      { status: 500 }
    );
  }
}
