import { NextRequest, NextResponse } from 'next/server';

// Backend URL - you can set this in environment variables
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

export async function POST(request: NextRequest) {
  try {
    // Parse the incoming request
    const body = await request.json();
    const { message, chatId, userId } = body;

    // Validate required fields
    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    try {
      // Try to forward the request to your Python backend
      const backendResponse = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          chatId,
          userId,
          timestamp: new Date().toISOString(),
        }),
      });

      // Check if backend request was successful
      if (!backendResponse.ok) {
        throw new Error('Backend service unavailable');
      }

      // Parse backend response
      const data = await backendResponse.json();

      // Return the response from backend
      return NextResponse.json({
        success: true,
        response: data.response || data.message,
        chatId: data.chatId || chatId,
        timestamp: data.timestamp || new Date().toISOString(),
      });

    } catch (backendError) {
      console.log('Backend unavailable, using mock response');
      
      // Generate mock AI response when backend is unavailable
      const mockResponses = [
        `I understand your question about "${message}". As an AI assistant, I'm here to help you with various tasks and provide information.`,
        `That's an interesting point about "${message}". Let me provide you with a comprehensive response.`,
        `Thank you for asking about "${message}". I can help you with that. Here's what I know about your inquiry.`,
        `I see you're asking about "${message}". This is a topic I can definitely assist you with.`,
        `Your question regarding "${message}" is quite relevant. I'll do my best to provide you with accurate information.`
      ];

      const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];

      return NextResponse.json({
        success: true,
        response: randomResponse,
        chatId: chatId || `chat_${Date.now()}`,
        timestamp: new Date().toISOString(),
      });
    }

  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Handle GET requests for chat history
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const chatId = searchParams.get('chatId');
    const userId = searchParams.get('userId');

    // Forward the request to your Python backend
    const backendResponse = await fetch(
      `${BACKEND_URL}/api/chat/history?chatId=${chatId}&userId=${userId}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!backendResponse.ok) {
      return NextResponse.json(
        { error: 'Failed to fetch chat history' },
        { status: 502 }
      );
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
