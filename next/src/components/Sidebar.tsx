"use client";

import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { supabaseChatService, Chat } from "@/services/supabaseChatService";

interface SidebarProps {
  onChatSelect?: (chatId: string) => void;
  currentPage?: "main" | "dashboard";
  isOpen: boolean;
  onToggle: () => void;
}

export default function Sidebar({ 
  onChatSelect, 
  currentPage = "main", 
  isOpen, 
  onToggle 
}: SidebarProps) {
  const router = useRouter();
  const [chats, setChats] = useState<Chat[]>([]);

  // Load chats from Supabase on component mount
  useEffect(() => {
    const loadChats = async () => {
      try {
        const loadedChats = await supabaseChatService.getChats();
        setChats(loadedChats);
      } catch (error) {
        console.error("Error loading chats:", error);
      }
    };

    loadChats();
  }, []);

  const handleChatSelect = (chatId: string) => {
    if (onChatSelect) {
      onChatSelect(chatId);
    } else {
      // Default behavior for dashboard - save to localStorage and navigate
      localStorage.setItem("selectedChatId", chatId);
      router.push("/");
    }
  };

  const testBackend = async () => {
    console.log("🔍 Testing LLM backend connection...");
    const result = await supabaseChatService.testLLMBackend();

    if (result.success) {
      console.log("✅ LLM Backend test successful:", result.message);
    } else {
      console.log("❌ LLM Backend test failed:", result.message);
    }

    alert(result.message);
  };

  return (
    <>
      {/* Left Sidebar - Fixed width with icons */}
      <div className="w-16 bg-black flex flex-col items-center py-6 space-y-8 relative z-10">
        {/* Menu Icon - Toggle Sidebar */}
        <button
          onClick={onToggle}
          className="p-3 text-white hover:text-gray-300 transition-colors"
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 6h16M4 12h16M4 18h16"
            />
          </svg>
        </button>

        {/* Dashboard Icon */}
        <button
          onClick={() => router.push("/dashboard")}
          className={`p-3 text-white hover:text-gray-300 transition-colors ${
            currentPage === "dashboard" ? "bg-gray-800 rounded" : ""
          }`}
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        </button>

        {/* Spacer */}
        <div className="flex-1"></div>

        {/* Settings Icon */}
        <button
          onClick={currentPage === "main" ? testBackend : undefined}
          className="p-3 text-white hover:text-gray-300 transition-colors"
          title={currentPage === "main" ? "Test LLM Backend Connection" : "Settings"}
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        </button>
      </div>

      {/* Chat History Sidebar - Toggleable */}
      {isOpen && (
        <div className="w-64 bg-gray-900 border-r border-gray-700 flex flex-col relative z-20 h-full">
          <div className="p-4 border-b border-gray-700">
            <button
              onClick={() => router.push("/")}
              className="text-white hover:text-gray-300 transition-colors text-lg font-light"
            >
              Chats
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <div className="space-y-2">
              {chats.length > 0 ? (
                chats.map((chat) => (
                  <div
                    key={chat.id}
                    onClick={() => handleChatSelect(chat.id)}
                    className="p-3 bg-gray-800 rounded cursor-pointer hover:bg-gray-700 transition-colors"
                  >
                    <div className="text-white text-sm">{chat.title}</div>
                    <div className="text-gray-400 text-xs">
                      {new Date(chat.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-gray-400 text-sm text-center py-4">
                  {currentPage === "main" 
                    ? "No chats yet. Start a conversation!" 
                    : "No chats yet"
                  }
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
