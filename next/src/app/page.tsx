"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from 'next/navigation';
import { supabaseChatService, ChatMessage, Chat } from "@/services/supabaseChatService";

// Using types from supabaseChatService
type Message = ChatMessage;

export default function Home() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [currentView, setCurrentView] = useState<'main' | 'chat' | 'dashboard'>('main');
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Typing animation state
  const [displayText, setDisplayText] = useState("");
  const [isTyping, setIsTyping] = useState(true);
  const [isDeleting, setIsDeleting] = useState(false);



  const createNewChat = async (title: string = '') => {
    try {
      const chatTitle = title || `Chat ${chats.length + 1}`;
      const newChat = await supabaseChatService.createChat(chatTitle);
      
      const updatedChats = [...chats, newChat];
      setChats(updatedChats);
      
      console.log('Created new chat and saved to Supabase:', newChat);
      setCurrentChatId(newChat.id);
      setMessages([]);
      
      return newChat.id;
    } catch (error) {
      console.error('Error creating new chat:', error);
      // Fallback to local ID generation if Supabase fails
      const fallbackId = supabaseChatService.generateChatId();
      setCurrentChatId(fallbackId);
      setMessages([]);
      return fallbackId;
    }
  };

  const selectChat = (chatId: string) => {
    const chat = chats.find(c => c.id === chatId);
    if (chat) {
      setCurrentChatId(chatId);
      setMessages(chat.messages);
      setCurrentView('chat');
      setIsSidebarOpen(false); // Automatically close sidebar when selecting a chat
    }
  };

  // Load chats from Supabase on component mount
  useEffect(() => {
    const loadChats = async () => {
      console.log('Component mounted, loading chats from Supabase...');
      
      // Test Supabase connection
      const isConnected = await supabaseChatService.testConnection();
      if (!isConnected) {
        console.log('💾 Using localStorage mode - chats will save locally');
        // Continue with localStorage fallback - don't return
      } else {
        console.log('🗄️ Connected to Supabase database');
      }

      try {
        const loadedChats = await supabaseChatService.getChats();
        setChats(loadedChats);
        console.log('✅ Successfully loaded chats:', loadedChats.length, 'chats found');

        // Check if returning from dashboard with a selected chat
        const selectedChatId = localStorage.getItem('selectedChatId');
        if (selectedChatId) {
          localStorage.removeItem('selectedChatId');
          const chat = loadedChats.find(c => c.id === selectedChatId);
          if (chat) {
            selectChat(selectedChatId);
          }
        }
      } catch (error) {
        console.log('ℹ️ Chat loading completed with fallback mode:', error);
      }
    };

    loadChats();
  }, []);

  // Typing animation effect
  useEffect(() => {
    const fullText = "Welcome to SLiM, where energy savings power higher-quality results.";
    const typingSpeed = 50; // milliseconds per character
    const deletingSpeed = 30; // milliseconds per character when deleting
    const pauseDuration = 2000; // 2 seconds pause
    
    let timeoutId: NodeJS.Timeout;
    
    if (isTyping && !isDeleting) {
      // Typing phase
      if (displayText.length < fullText.length) {
        timeoutId = setTimeout(() => {
          setDisplayText(fullText.slice(0, displayText.length + 1));
        }, typingSpeed);
      } else {
        // Finished typing, pause then start deleting
        timeoutId = setTimeout(() => {
          setIsDeleting(true);
          setIsTyping(false);
        }, pauseDuration);
      }
    } else if (isDeleting && !isTyping) {
      // Deleting phase
      if (displayText.length > 0) {
        timeoutId = setTimeout(() => {
          setDisplayText(displayText.slice(0, -1));
        }, deletingSpeed);
      } else {
        // Finished deleting, start typing again
        setIsDeleting(false);
        setIsTyping(true);
      }
    }
    
    return () => clearTimeout(timeoutId);
  }, [displayText, isTyping, isDeleting]);

  const returnToMainScreen = () => {
    setCurrentChatId(null);
    setMessages([]);
    setCurrentView('main');
    setIsSidebarOpen(false); // Automatically close sidebar when returning to main screen
  };

  const showDashboard = () => {
    // Navigate to dashboard route (chats are now in Supabase)
    router.push('/dashboard');
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const currentInput = inputValue;
    setInputValue("");
    setIsLoading(true);

    try {
      // Create new chat if none exists
      let chatId = currentChatId;
      if (!chatId) {
        chatId = await createNewChat(currentInput.length > 30 ? currentInput.substring(0, 30) + "..." : currentInput);
      }

      setCurrentView('chat');

      // Send message and get response through Supabase service
      const response = await supabaseChatService.sendMessage(currentInput, chatId);
      
      if (response.success) {
        // Reload chat to get updated messages from database
        const updatedChat = await supabaseChatService.getChat(chatId);
        if (updatedChat) {
          setMessages(updatedChat.messages);
          
          // Update the chat in our local state
          setChats(prevChats => {
            const existingChatIndex = prevChats.findIndex(chat => chat.id === chatId);
            if (existingChatIndex >= 0) {
              // Update existing chat
              const newChats = [...prevChats];
              newChats[existingChatIndex] = updatedChat;
              return newChats;
            } else {
              // Add new chat
              return [...prevChats, updatedChat];
            }
          });
        }
        
        console.log('Message sent successfully to Supabase:', response);
      } else {
        console.error('Failed to send message:', response.error);
        // The service already handles fallback responses, so we don't need to do anything here
      }
      
    } catch (error) {
      console.error('Error in handleSendMessage:', error);
      
      // Show local error message as fallback
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        text: "Sorry, I encountered an error. Please try again.",
        isUser: false,
        timestamp: new Date().toISOString(),
        chatId: currentChatId || '',
      };
      setMessages(prev => [...prev, errorMessage]);
      
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleAttachFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      console.log("File selected:", file.name);
      
      try {
        setIsLoading(true);
        const result = await supabaseChatService.uploadFile(file, currentChatId || undefined);
        
        if (result.success) {
          // Add a message about the uploaded file
          const fileMessage: Message = {
            id: Date.now().toString(),
            text: `📎 Uploaded file: ${file.name}`,
            isUser: true,
            timestamp: new Date().toISOString(),
            chatId: currentChatId || '',
          };
          
          setMessages(prev => [...prev, fileMessage]);
          
          // Update chat with file message
          if (currentChatId) {
            setChats(prev => prev.map(chat => 
              chat.id === currentChatId 
                ? { ...chat, messages: [...chat.messages, fileMessage] }
                : chat
            ));
          }
        }
      } catch (error) {
        console.error('File upload error:', error);
        // Could show an error toast here
      } finally {
        setIsLoading(false);
        // Clear the file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    }
  };

  return (
    <div className="h-screen bg-black flex relative overflow-hidden">
      {/* Frame Elements - Only show on welcome screen */}
      {currentView === 'main' && (
        <div className="absolute inset-8 pointer-events-none">
          {/* L-shaped corners - only show on welcome screen */}
          <div className="absolute top-0 left-13 w-16 h-16 border-l-3 border-t-2 border-white"></div>
          <div className="absolute bottom-0 right-0 w-16 h-16 border-r-2 border-b-2 border-white"></div>
          <div className="absolute top-0 right-0 w-16 h-16 border-r-2 border-t-2 border-white"></div>
          <div className="absolute bottom-0 left-13 w-16 h-16 border-l-2 border-b-2 border-white"></div>
        </div>
      )}
      
      {/* SLIM Text with dropdown and 01 - only show on main screen */}
      {currentView === 'main' && (
        <div className="absolute top-10 left-23 pointer-events-none z-20">
          <div className="flex items-center space-x-2">
            <span className="text-white text-2xl font-light tracking-wider font-sans">SLiM</span>
            <span className="text-gray-500 text-2xl font-light font-sans">01</span>
            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      )}

      {/* Left Sidebar - Hide when in dashboard view */}
      {currentView !== 'dashboard' && (
        <div className="w-16 bg-black flex flex-col items-center py-6 space-y-8 relative z-10">
        {/* Menu Icon - Toggle Sidebar */}
        <button 
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="p-3 text-white hover:text-gray-300 transition-colors"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        
        {/* Dashboard Icon */}
        <button 
          onClick={showDashboard}
          className="p-3 text-white hover:text-gray-300 transition-colors"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </button>

        {/* Spacer */}
        <div className="flex-1"></div>

        {/* Settings Icon */}
        <button className="p-3 text-white hover:text-gray-300 transition-colors">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>
        </div>
      )}

      {/* White separator line - Hide when in dashboard view */}
      {currentView !== 'dashboard' && (
        <div className="flex flex-col items-center justify-center relative z-10 h-full py-16">
          {/* Middle separator line with padding */}
          <div className="w-px bg-white flex-1"></div>
        </div>
      )}
      
      {/* Independent white squares positioned between corners - Hide when in dashboard view */}
      {currentView !== 'dashboard' && (
        <div className="absolute inset-0 pointer-events-none z-10">
        {/* Top square - between top corners, centered horizontally */}
        <div className="absolute top-1/2 left-1/10 transform -translate-x-1/2 w-3 h-3 bg-white"></div>
        
        {/* Bottom square - between bottom corners, centered horizontally */}
        <div className="absolute bottom-1/2 right-1 transform -translate-x-1/2 w-3 h-3 bg-white"></div>
        
        {/* Extending lines from squares - only show in chat screen */}
        {messages.length > 0 && (
          <>
            {/* Lines extending from top square */}
            <div className="absolute top-1/2 left-1/10 transform -translate-x-1/2">
              {/* Left extending line from top square */}
              <div className="absolute bottom-1/2 right-0 w-px h-40 bg-white transform -translate-y-1/2"></div>
              {/* Right extending line from top square */}
              <div className="absolute top-40 left-0 w-px h-40 bg-white transform -translate-y-1/2"></div>
            </div>
            
            {/* Lines extending from bottom square */}
            <div className="absolute bottom-1/2 right-1/20 transform -translate-x-1/2">
              {/* Left extending line from bottom square */}
              <div className="absolute -top-40 -right-7 w-px h-40 bg-white transform -translate-y-1/2"></div>
              {/* Right extending line from bottom square */}
              <div className="absolute top-40 left-6.5 w-px h-40 bg-white transform -translate-y-1/2"></div>
            </div>
          </>
        )}
        </div>
      )}
    
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Main Content */}
        <main className="flex-1 flex flex-col relative z-10">
        {currentView === 'main' ? (
          /* Welcome Screen */
          <div className="flex h-full">
            {/* Chat History Sidebar - Also available on main screen */}
            {isSidebarOpen && (
              <div className="w-64 bg-gray-800 p-4 space-y-2 mt-16">
                <div className="text-white text-lg font-medium mb-4">Chats</div>
                <div className="space-y-1">
                  {chats.length > 0 ? (
                    chats.map((chat) => (
                      <div 
                        key={chat.id}
                        onClick={() => selectChat(chat.id)}
                        className="p-3 text-gray-400 text-sm cursor-pointer hover:bg-gray-700 hover:text-white rounded transition-colors"
                      >
                        {chat.title}
                      </div>
                    ))
                  ) : (
                    <div className="p-3 text-gray-500 text-sm text-center">
                      No chats yet. Start a conversation!
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Main Welcome Content */}
            <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
              <div className="mb-16">
                <div className="text-2xl font-light text-white leading-relaxed min-h-[4rem] flex items-center justify-center">
                  <span className="inline-block">
                    {displayText}
                    <span className="animate-pulse text-white">|</span>
                  </span>
                </div>
              </div>
              
              {/* Input Area */}
              <div className="w-full max-w-2xl">
                <div className="relative">
                  <div className="flex items-center bg-white rounded-2xl p-4 shadow-lg">
                    {/* Search Icon */}
                    <svg className="w-5 h-5 text-gray-400 mr-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    
                    <input
                      type="text"
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask SLiM anything..."
                      className="flex-1 bg-transparent text-gray-800 placeholder-gray-500 focus:outline-none text-lg"
                    />
                    
                    {/* Attach Button */}
                    <button
                      onClick={handleAttachFile}
                      className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors border border-gray-300 rounded-full ml-4"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                      </svg>
                      <span className="text-sm">Attach</span>
                    </button>
                    
                    {/* Send Button */}
                    <button
                      onClick={handleSendMessage}
                      disabled={!inputValue.trim() || isLoading}
                      className="ml-4 p-2 text-gray-600 hover:text-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
            /* Chat Messages */
            <div className="flex h-full">
              {/* Chat History Sidebar - Toggleable */}
              {isSidebarOpen && (
                <div className="w-64 bg-gray-800 p-4 space-y-2 mt-16">
                  <button 
                    onClick={returnToMainScreen}
                    className="text-white text-lg font-medium mb-4 hover:text-gray-300 cursor-pointer"
                  >
                    Chats
                  </button>
                  <div className="space-y-1">
                    {chats.map((chat) => (
                      <div 
                        key={chat.id}
                        onClick={() => selectChat(chat.id)}
                        className={`p-3 text-sm cursor-pointer rounded transition-colors ${
                          currentChatId === chat.id 
                            ? 'bg-gray-700 text-white' 
                            : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                        }`}
                      >
                        {chat.title}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Main Chat Area */}
              <div className="flex-1 flex flex-col">
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                  {messages.map((message) => (
                    <div key={message.id} className="space-y-4">
                      {message.isUser ? (
                        /* User Message with top-right corner arrow only */
                        <div className="flex justify-end pr-8">
                          <div className="max-w-lg bg-transparent right-5 p-4 text-white text-sm leading-relaxed relative">
                            {/* Top-right corner arrow indicator */}
                            <div className="absolute top-1 right-1 w-12 h-12 border-t-2 border-r-2 border-white"></div>
                            {message.text}
                          </div>
                        </div>
                      ) : (
                        /* AI Message with top-left corner arrow only */
                        <div className="flex justify-start pl-8">
                          <div className="max-w-lg bg-transparent left-11 p-4 text-white text-sm leading-relaxed relative">
                            {/* Top-left corner arrow indicator */}
                            <div className="absolute top-1 left-2 w-12 h-12 border-t-2 border-l-2 border-white"></div>
                            {message.text}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                  {isLoading && (
                    <div className="flex justify-start pl-8">
                      <div className="max-w-lg bg-transparent p-4 relative">
                        {/* Top-left corner arrow indicator for loading */}
                        <div className="absolute top-1 left-2 w-12 h-12 border-t-2 border-l-2 border-white"></div>
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Input Area for Chat Mode */}
                <div className="p-6">
                  <div className="relative">
                    <div className="flex items-center bg-white rounded-2xl p-4 shadow-lg">
                      {/* Search Icon */}
                      <svg className="w-5 h-5 text-gray-400 mr-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      
                      <input
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask SLiM anything..."
                        className="flex-1 bg-transparent text-gray-800 placeholder-gray-500 focus:outline-none text-lg"
                      />
                      
                      {/* Attach Button */}
                      <button
                        onClick={handleAttachFile}
                        className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors border border-gray-300 rounded-full ml-4"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                        </svg>
                        <span className="text-sm">Attach</span>
                      </button>
                      
                      {/* Send Button */}
                      <button
                        onClick={handleSendMessage}
                        disabled={!inputValue.trim() || isLoading}
                        className="ml-4 p-2 text-gray-600 hover:text-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        onChange={handleFileChange}
        className="hidden"
        accept="*/*"
      />
    </div>
  );
}
