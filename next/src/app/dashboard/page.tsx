"use client";

import { useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import { supabaseChatService, Chat } from '@/services/supabaseChatService';

// Using Chat interface from supabaseChatService

interface MindMapNode {
  id: string;
  label: string;
  sublabel: string;
  percentage: number;
  color: string;
  position: { x: string; y: string };
}

export default function Dashboard() {
  const router = useRouter();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [chats, setChats] = useState<Chat[]>([]);
  const [showProcessingModal, setShowProcessingModal] = useState(false);
  const [hasNvidiaGpu, setHasNvidiaGpu] = useState<boolean | null>(null);
  const [manualGpuOverride, setManualGpuOverride] = useState(false);
  const [processingType, setProcessingType] = useState<'cloud' | 'gpu'>('cloud');
  const [processingStatus, setProcessingStatus] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [nodes, setNodes] = useState<MindMapNode[]>([
    // Core Training Nodes - Better distributed
    {
      id: 'training-3',
      label: 'Training',
      sublabel: 'set 3',
      percentage: 98,
      color: 'from-green-400 via-green-500 to-emerald-600',
      position: { x: '20%', y: '15%' }
    },
    {
      id: 'training-2',
      label: 'Training',
      sublabel: 'set 2',
      percentage: 99,
      color: 'from-blue-400 via-blue-500 to-indigo-600',
      position: { x: '80%', y: '15%' }
    },
    {
      id: 'training-1',
      label: 'Training',
      sublabel: 'set 1',
      percentage: 96,
      color: 'from-purple-400 via-purple-500 to-violet-600',
      position: { x: '15%', y: '80%' }
    },
    
    // Processing & Analysis - More spacing
    {
      id: 'processing',
      label: 'Process',
      sublabel: 'Node',
      percentage: 85,
      color: 'from-orange-400 via-red-500 to-pink-600',
      position: { x: '85%', y: '80%' }
    },
    {
      id: 'analysis',
      label: 'Analysis',
      sublabel: 'Engine',
      percentage: 92,
      color: 'from-cyan-400 via-teal-500 to-blue-600',
      position: { x: '50%', y: '8%' }
    },
    
    // Neural Network Components - Better spacing
    {
      id: 'neural-net',
      label: 'Neural',
      sublabel: 'Network',
      percentage: 94,
      color: 'from-indigo-400 via-purple-500 to-pink-600',
      position: { x: '8%', y: '45%' }
    },
    {
      id: 'transformer',
      label: 'Transform',
      sublabel: 'Layer',
      percentage: 97,
      color: 'from-teal-400 via-cyan-500 to-blue-600',
      position: { x: '92%', y: '45%' }
    },
    
    // Data Processing - More spread out
    {
      id: 'tokenizer',
      label: 'Token',
      sublabel: 'Engine',
      percentage: 89,
      color: 'from-yellow-400 via-orange-500 to-red-600',
      position: { x: '30%', y: '90%' }
    },
    {
      id: 'embeddings',
      label: 'Embed',
      sublabel: 'Vector',
      percentage: 91,
      color: 'from-rose-400 via-pink-500 to-purple-600',
      position: { x: '70%', y: '90%' }
    },
    
    // Memory & Storage - Edge positions
    {
      id: 'memory-bank',
      label: 'Memory',
      sublabel: 'Bank',
      percentage: 88,
      color: 'from-emerald-400 via-teal-500 to-cyan-600',
      position: { x: '5%', y: '25%' }
    },
    {
      id: 'cache-layer',
      label: 'Cache',
      sublabel: 'Layer',
      percentage: 95,
      color: 'from-violet-400 via-purple-500 to-indigo-600',
      position: { x: '95%', y: '25%' }
    },
    
    // Optimization & Performance - Mid-range spacing
    {
      id: 'optimizer',
      label: 'Optim',
      sublabel: 'Engine',
      percentage: 93,
      color: 'from-lime-400 via-green-500 to-emerald-600',
      position: { x: '25%', y: '65%' }
    },
    {
      id: 'scheduler',
      label: 'Schedule',
      sublabel: 'Manager',
      percentage: 87,
      color: 'from-amber-400 via-yellow-500 to-orange-600',
      position: { x: '75%', y: '65%' }
    },
    
    // Security & Monitoring - Corner positions
    {
      id: 'security',
      label: 'Security',
      sublabel: 'Guard',
      percentage: 99,
      color: 'from-red-400 via-rose-500 to-pink-600',
      position: { x: '10%', y: '65%' }
    },
    {
      id: 'monitor',
      label: 'Monitor',
      sublabel: 'System',
      percentage: 90,
      color: 'from-sky-400 via-blue-500 to-indigo-600',
      position: { x: '90%', y: '65%' }
    },
    
    // Advanced Features - Central and balanced
    {
      id: 'attention',
      label: 'Attention',
      sublabel: 'Mechanism',
      percentage: 96,
      color: 'from-fuchsia-400 via-purple-500 to-violet-600',
      position: { x: '50%', y: '92%' }
    },
    {
      id: 'inference',
      label: 'Inference',
      sublabel: 'Engine',
      percentage: 94,
      color: 'from-cyan-400 via-blue-500 to-purple-600',
      position: { x: '35%', y: '30%' }
    },
    {
      id: 'fine-tune',
      label: 'Fine-tune',
      sublabel: 'Module',
      percentage: 86,
      color: 'from-green-400 via-emerald-500 to-teal-600',
      position: { x: '65%', y: '30%' }
    }
  ]);

  // Load chats from Supabase on component mount
  useEffect(() => {
    const loadChats = async () => {
      try {
        const loadedChats = await supabaseChatService.getChats();
        setChats(loadedChats);
        console.log('Dashboard: Loaded chats from Supabase:', loadedChats);
      } catch (error) {
        console.error('Dashboard: Error loading chats from Supabase:', error);
      }
    };

    loadChats();
  }, []);

  // Simulate API data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setNodes(prevNodes => 
        prevNodes.map(node => ({
          ...node,
          percentage: Math.max(70, Math.min(100, node.percentage + (Math.random() - 0.5) * 4))
        }))
      );
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // Calculate node size based on percentage (bigger nodes for better visibility)
  const getNodeSize = (percentage: number) => {
    const minSize = 19; // 3rem = 48px
    const maxSize = 25; // 4.5rem = 72px
    const size = minSize + ((percentage - 70) / 30) * (maxSize - minSize);
    return Math.max(minSize, Math.min(maxSize, size));
  };

  // Enhanced GPU Detection Function
  const detectNvidiaGpu = async (): Promise<boolean> => {
    try {
      // Method 1: Try WebGL detection first
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
          const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
          
          console.log('WebGL GPU Renderer:', renderer);
          console.log('WebGL GPU Vendor:', vendor);

          // Check for NVIDIA in renderer or vendor strings
          const webglHasNvidia = renderer.toLowerCase().includes('nvidia') || 
                                vendor.toLowerCase().includes('nvidia') ||
                                renderer.toLowerCase().includes('geforce') ||
                                renderer.toLowerCase().includes('quadro') ||
                                renderer.toLowerCase().includes('tesla') ||
                                renderer.toLowerCase().includes('rtx') ||
                                renderer.toLowerCase().includes('gtx');

          if (webglHasNvidia) {
            console.log('✅ NVIDIA GPU detected via WebGL');
            return true;
          }
        }
      }

      // Method 2: Try to check via backend API (if available)
      try {
        const response = await fetch('http://localhost:5001/cuda/check', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('Backend GPU check:', data);
          if (data.cuda_available || data.nvidia_gpu) {
            console.log('✅ NVIDIA GPU detected via backend');
            return true;
          }
        }
      } catch (backendError) {
        console.log('Backend GPU check unavailable:', backendError);
      }

      // Method 3: Check navigator.gpu (WebGPU) if available
      if ('gpu' in navigator) {
        try {
          const adapter = await (navigator as any).gpu.requestAdapter();
          if (adapter) {
            const info = await adapter.requestAdapterInfo();
            console.log('WebGPU Adapter Info:', info);
            
            if (info.vendor?.toLowerCase().includes('nvidia') || 
                info.description?.toLowerCase().includes('nvidia') ||
                info.description?.toLowerCase().includes('geforce') ||
                info.description?.toLowerCase().includes('rtx') ||
                info.description?.toLowerCase().includes('gtx')) {
              console.log('✅ NVIDIA GPU detected via WebGPU');
              return true;
            }
          }
        } catch (webgpuError) {
          console.log('WebGPU check failed:', webgpuError);
        }
      }

      console.log('❌ No NVIDIA GPU detected via any method');
      return false;
    } catch (error) {
      console.error('Error detecting GPU:', error);
      return false;
    }
  };

  // Cloud Processing Function (current implementation)
  const runCloudProcessing = async () => {
    setProcessingStatus('Initializing cloud processing...');
    setIsProcessing(true);

    try {
      // Simulate cloud processing steps (replace with actual cloud API calls)
      setProcessingStatus('Connecting to cloud services...');
      await new Promise(resolve => setTimeout(resolve, 1000));

      setProcessingStatus('Processing with cloud LLM...');
      await new Promise(resolve => setTimeout(resolve, 2000));

      setProcessingStatus('Generating results...');
      await new Promise(resolve => setTimeout(resolve, 1500));

      setProcessingStatus('Cloud processing completed successfully!');
      
      // Navigate to main chat after successful processing
      setTimeout(() => {
        router.push('/');
      }, 1000);

    } catch (error) {
      console.error('Cloud processing error:', error);
      setProcessingStatus('Cloud processing failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // GPU Processing Function (new CUDA implementation)
  const runGpuProcessing = async () => {
    setProcessingStatus('Initializing NVIDIA GPU processing...');
    setIsProcessing(true);

    try {
      // Step 1: Initialize CUDA environment
      setProcessingStatus('Checking CUDA availability...');
      const cudaCheckResponse = await fetch('http://localhost:5001/cuda/check', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!cudaCheckResponse.ok) {
        throw new Error('CUDA not available on system');
      }

      // Step 2: Run CUDA initialization (cudaInit.py)
      setProcessingStatus('Running CUDA initialization...');
      const initResponse = await fetch('http://localhost:5001/cuda/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          base_model: 'gpt2',
          data_path: '../UserFacing/db/LLMData.json',
          output_dir: './cuda_lora_out'
        })
      });

      if (!initResponse.ok) {
        throw new Error('CUDA initialization failed');
      }

      // Step 3: Run test suite (testSuite.py)
      setProcessingStatus('Running CUDA test suite...');
      const testResponse = await fetch('http://localhost:5001/cuda/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_model: 'gpt2',
          lora_model_path: './cuda_lora_out'
        })
      });

      if (!testResponse.ok) {
        throw new Error('CUDA test suite failed');
      }

      const testResults = await testResponse.json();
      setProcessingStatus('GPU processing completed successfully!');
      
      console.log('GPU Processing Results:', testResults);
      
      // Navigate to main chat after successful processing
      setTimeout(() => {
        router.push('/');
      }, 1000);

    } catch (error) {
      console.error('GPU processing error:', error);
      setProcessingStatus(`GPU processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle processing type selection
  const handleProcessingStart = async () => {
    setShowProcessingModal(false);
    
    if (processingType === 'gpu') {
      await runGpuProcessing();
    } else {
      await runCloudProcessing();
    }
  };

  // Handle main button click
  const handleMainButtonClick = async () => {
    // Detect GPU if not already detected
    if (hasNvidiaGpu === null) {
      setProcessingStatus('Detecting NVIDIA GPU...');
      const gpuDetected = await detectNvidiaGpu();
      setHasNvidiaGpu(gpuDetected);
    }

    // Show modal for processing type selection
    setShowProcessingModal(true);
  };

  const handleChatSelect = (chatId: string) => {
    // Save selected chat to localStorage and navigate
    localStorage.setItem('selectedChatId', chatId);
    router.push('/');
  };
  
  return (
    <div className="h-screen bg-black flex relative overflow-hidden">
      {/* Sidebar */}
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
        
        {/* Dashboard Icon - Currently Active */}
        <button 
          className="p-3 text-white hover:text-gray-300 transition-colors bg-gray-800 rounded"
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

      {/* Chat History Sidebar - Toggleable */}
      {isSidebarOpen && (
        <div className="w-64 bg-gray-900 border-r border-gray-700 flex flex-col relative z-20 h-full">
          <div className="p-4 border-b border-gray-700">
            <button 
              onClick={() => router.push('/')}
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
                  No chats yet
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Main Content Container */}
      <div className="flex-1 flex relative">
        {/* SLIM Header - Clickable */}
        <div className="absolute top-10 left-8 z-30">
          <button 
            onClick={() => router.push('/')}
            className="flex items-center space-x-2 hover:opacity-80 transition-opacity cursor-pointer"
          >
            <span className="text-white text-2xl font-light tracking-wider font-sans">SLiM</span>
            <span className="text-gray-500 text-2xl font-light font-sans">01</span>
            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>

        {/* Corner brackets positioned at specific screen locations */}
        <div className="absolute inset-0 pointer-events-none z-10">
          {/* Top-right corner bracket */}
          <div className="absolute top-8 right-8 w-16 h-16 border-r-2 border-t-2 border-white"></div>
          {/* Bottom-right corner bracket */}
          <div className="absolute bottom-8 right-8 w-16 h-16 border-r-2 border-b-2 border-white"></div>
          {/* Middle-top corner bracket */}
          <div className="absolute top-8 left-180 transform -translate-x-1/2 w-16 h-16 border-l-2 border-t-2 border-white"></div>
          {/* Middle-bottom corner bracket */}
          <div className="absolute bottom-8 left-180 transform -translate-x-1/2 w-16 h-16 border-l-2 border-b-2 border-white"></div>
        </div>

        {/* Left Content Area */}
        <div className="w-1/2 pt-8 pr-4 pl-8 relative z-20 flex flex-col justify-center h-full">
        {/* Main Section - Centered */}
        <div className="flex flex-col justify-center flex-1">
          {/* Dashboard Header */}
          <div className="mb-6">
            <h2 className="text-xl font-light text-red-400 mb-4">RUN OPEN-SOURCE MODEL •</h2>
            <p className="text-gray-300 text-sm mb-1">Save Power by making ur LLM's into SLiM's and</p>
            <p className="text-gray-300 text-sm mb-6">producing equal to better results</p>
          </div>

          {/* Action Button */}
          <div className="mb-8">
            <div className="relative inline-block">
              {/* Corner brackets */}
              <div className="absolute -top-1 -left-1 w-4 h-4 border-l-2 border-t-2 border-white"></div>
              <div className="absolute -top-1 -right-1 w-4 h-4 border-r-2 border-t-2 border-white"></div>
              <div className="absolute -bottom-1 -left-1 w-4 h-4 border-l-2 border-b-2 border-white"></div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 border-r-2 border-b-2 border-white"></div>
              <button 
                onClick={handleMainButtonClick}
                className="px-6 py-3 text-base font-light cursor-pointer hover:bg-white hover:text-black transition-colors"
                disabled={isProcessing}
              >
                {isProcessing ? 'Processing...' : 'Start AI Processing'}
              </button>
            </div>
          </div>

          {/* Processing Status */}
          {processingStatus && (
            <div className="mb-6">
              <div className="bg-gray-800/50 border border-gray-600 rounded-lg p-4">
                <div className="text-white text-sm">{processingStatus}</div>
                {isProcessing && (
                  <div className="mt-2">
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Progress Indicators with brackets */}
          <div className="mb-8">
            <div className="flex items-center">
              <span className="text-white mr-2">[</span>
              <div className="flex items-center space-x-1">
                {/* Section 1 */}
                {Array.from({ length: 4 }, (_, i) => (
                  <div key={i} className="w-3 h-3 bg-gray-400"></div>
                ))}
                
                {/* Gap */}
                <div className="w-4"></div>
                
                {/* Section 2 */}
                {Array.from({ length: 7 }, (_, i) => (
                  <div key={i + 8} className="w-3 h-3 bg-gray-400"></div>
                ))}
                
                {/* Gap */}
                <div className="w-4"></div>
                
                {/* Section 3 */}
                {Array.from({ length: 6 }, (_, i) => (
                  <div key={i + 15} className="w-3 h-3 bg-gray-400"></div>
                ))}
                
                {/* Gap */}
                <div className="w-4"></div>
                
                {/* Section 4 */}
                {Array.from({ length: 8 }, (_, i) => (
                  <div key={i + 21} className="w-3 h-3 bg-gray-400"></div>
                ))}
                
                {/* Gap */}
                <div className="w-4"></div>
                
                {/* Section 5 */}
                {Array.from({ length: 8 }, (_, i) => (
                  <div key={i + 29} className="w-3 h-3 bg-gray-400"></div>
                ))}
              </div>
              <span className="text-white ml-2">]</span>
            </div>
          </div>

          {/* Resource Overview and Parameters */}
          <div className="grid grid-cols-2 gap-8 mb-6">
            {/* Left side - Resource Overview */}
            <div>
              <h3 className="text-xl font-light mb-2">Resource</h3>
              <h3 className="text-xl font-light">Overview</h3>
            </div>
            
            {/* Right side - Parameters with bracket styling */}
            <div className="space-y-3 text-base">
              <div className="flex items-center">
                <span className="text-white">[</span>
                <span className="text-gray-400 mx-2">Parameters:</span>
                <span className="text-white">2B</span>
                <span className="text-white ml-2">]</span>
              </div>
              <div className="flex items-center">
                <span className="text-white">[</span>
                <span className="text-gray-400 mx-2">Heat Saved:</span>
                <span className="text-white">10°C</span>
                <span className="text-white ml-2">]</span>
              </div>
              <div className="flex items-center">
                <span className="text-white">[</span>
                <span className="text-gray-400 mx-2">Token Saved:</span>
                <span className="text-white">$20</span>
                <span className="text-white ml-2">]</span>
              </div>
            </div>
          </div>

          {/* Analysis Section */}
          <div className="mt-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-light">ANALYSIS</h3>
            <select className="bg-gray-800 text-white px-2 py-1 text-sm border border-gray-600">
              <option>1 Day</option>
              <option>1 Week</option>
              <option>1 Month</option>
            </select>
          </div>
          <div className="flex items-center mb-4">
            <span className="text-white">[</span>
            <span className="text-red-400 mx-2">• Heat Usage</span>
            <span className="text-white">]</span>
          </div>
          
          {/* Chart */}
          <div className="bg-transparent p-3">
            <div className="flex">
              {/* Y-axis with temperature labels */}
              <div className="flex flex-col justify-between h-28 mr-2 text-xs text-gray-400">
                <span>100°C</span>
                <span>80°C</span>
                <span>60°C</span>
                <span>40°C</span>
                <span>20°C</span>
                <span>0°C</span>
              </div>
              
              {/* Chart bars */}
              <div className="flex-1">
                <div className="grid grid-cols-12 gap-1 h-28 items-end">
                  {[20, 35, 45, 30, 50, 40, 35, 45, 40, 25, 30, 35].map((height, i) => (
                    <div key={i} className="bg-red-500 rounded-t" style={{ height: `${height}%` }}></div>
                  ))}
                </div>
              </div>
            </div>
            <div className="flex justify-between text-xs text-gray-400 mt-1 ml-6">
              <span>00:00</span>
              <span>06:00</span>
              <span>12:00</span>
              <span>18:00</span>
              <span>24:00</span>
            </div>
          </div>
          </div>
        </div>
        </div>

        {/* White separator line */}
        <div className="flex flex-col items-center -left-175 justify-center relative z-10 h-full py-16">
          {/* Middle separator line with padding */}
          <div className="w-px bg-white flex-1"></div>
        </div>

        {/* Right Content Area */}
        <div className="w-1/2 pt-8 pr-8 pl-4 flex items-center justify-center relative z-20 h-full">
        {/* Mind Map Network */}
        <div className="relative w-full h-full flex items-center justify-center">
          {/* Connection Lines */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 1 }}>
            {/* Central hub connections to all nodes */}
            {nodes.map((node, index) => (
              <line 
                key={`hub-${node.id}`}
                x1="50%" 
                y1="50%" 
                x2={node.position.x} 
                y2={node.position.y} 
                stroke={`rgba(255,255,255,${0.2 + (node.percentage / 100) * 0.3})`} 
                strokeWidth={1 + (node.percentage / 100) * 2} 
                strokeDasharray="5,5"
              >
                <animate 
                  attributeName="stroke-dashoffset" 
                  values="0;10" 
                  dur={`${1.5 + index * 0.3}s`} 
                  repeatCount="indefinite"
                />
              </line>
            ))}
            
            {/* Inter-node connections - connect all nodes to each other */}
            {nodes.map((nodeA, indexA) => 
              nodes.slice(indexA + 1).map((nodeB, indexB) => (
                <line 
                  key={`${nodeA.id}-${nodeB.id}`}
                  x1={nodeA.position.x} 
                  y1={nodeA.position.y} 
                  x2={nodeB.position.x} 
                  y2={nodeB.position.y} 
                  stroke="rgba(255,255,255,0.08)" 
                  strokeWidth="0.5" 
                  strokeDasharray="2,4"
                >
                  <animate 
                    attributeName="stroke-dashoffset" 
                    values="0;6" 
                    dur={`${2 + (indexA + indexB) * 0.2}s`} 
                    repeatCount="indefinite"
                  />
                </line>
              ))
            )}
          </svg>

          {/* Central Hub Node - Cosmic Core */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" style={{ zIndex: 10 }}>
            <div className="relative group">
              {/* Main cosmic sphere */}
              <div className="relative w-20 h-20 rounded-full overflow-hidden shadow-2xl border border-white/20 hover:scale-105 transition-all duration-300"
                   style={{
                     background: `
                       radial-gradient(circle at 30% 20%, rgba(147, 51, 234, 0.3) 0%, transparent 70%),
                       radial-gradient(circle at 70% 80%, rgba(59, 130, 246, 0.2) 0%, transparent 60%),
                       linear-gradient(45deg, #1f2937 0%, #374151 50%, #111827 100%)
                     `
                   }}>
                
                {/* Subtle starfield background */}
                <div className="absolute inset-0">
                  {Array.from({ length: 8 }, (_, i) => (
                    <div
                      key={i}
                      className="absolute bg-white/40 rounded-full animate-pulse"
                      style={{
                        width: `${Math.random() * 1 + 0.5}px`,
                        height: `${Math.random() * 1 + 0.5}px`,
                        top: `${Math.random() * 100}%`,
                        left: `${Math.random() * 100}%`,
                        animationDelay: `${Math.random() * 5}s`,
                        animationDuration: `${3 + Math.random() * 3}s`
                      }}
                    />
                  ))}
                </div>
                
                {/* Subtle nebula swirl effect */}
                <div className="absolute inset-0 rounded-full opacity-30">
                  <div className="absolute inset-0 rounded-full animate-spin" style={{ animationDuration: '20s' }}>
                    <div className="absolute top-2 left-2 w-2 h-2 bg-gradient-to-br from-gray-400/20 to-transparent rounded-full blur-sm"></div>
                    <div className="absolute bottom-2 right-2 w-2 h-2 bg-gradient-to-br from-gray-400/20 to-transparent rounded-full blur-sm"></div>
                  </div>
                </div>
                
                {/* Central content */}
                <div className="relative z-10 flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="text-white text-xs font-light tracking-wider drop-shadow-lg">CORE</div>
                    <div className="text-white text-lg font-bold tracking-wide drop-shadow-lg">AI</div>
                  </div>
                </div>
                
                {/* Subtle energy pulse */}
                <div className="absolute inset-0 rounded-full bg-gradient-radial from-white/5 via-transparent to-transparent animate-pulse"></div>
              </div>
              
              {/* Subtle outer glow effect */}
              <div className="absolute inset-0 rounded-full bg-gradient-radial from-gray-400/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-lg scale-125"></div>
            </div>
          </div>

          {/* Dynamic Space Nodes */}
          {nodes.map((node, index) => {
            const size = getNodeSize(node.percentage);
            const nodeSize = size * 3; // Reduced multiplier for smaller nodes
            
            // Define subtle muted backgrounds for each node
            const spaceBackgrounds = {
              // Core Training Nodes
              'training-3': `
                radial-gradient(circle at 25% 25%, rgba(34, 197, 94, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 75% 75%, rgba(16, 185, 129, 0.2) 0%, transparent 50%),
                linear-gradient(135deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'training-2': `
                radial-gradient(circle at 30% 20%, rgba(59, 130, 246, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 80% 80%, rgba(99, 102, 241, 0.2) 0%, transparent 50%),
                linear-gradient(225deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'training-1': `
                radial-gradient(circle at 40% 30%, rgba(147, 51, 234, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 70% 70%, rgba(168, 85, 247, 0.2) 0%, transparent 50%),
                linear-gradient(315deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              
              // Processing & Analysis
              'processing': `
                radial-gradient(circle at 35% 25%, rgba(251, 146, 60, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 65% 75%, rgba(239, 68, 68, 0.2) 0%, transparent 50%),
                linear-gradient(45deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'analysis': `
                radial-gradient(circle at 20% 30%, rgba(6, 182, 212, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 80% 60%, rgba(14, 165, 233, 0.2) 0%, transparent 50%),
                linear-gradient(180deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              
              // Neural Network Components
              'neural-net': `
                radial-gradient(circle at 30% 40%, rgba(99, 102, 241, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 70% 20%, rgba(147, 51, 234, 0.2) 0%, transparent 50%),
                linear-gradient(180deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'transformer': `
                radial-gradient(circle at 40% 30%, rgba(20, 184, 166, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 60% 70%, rgba(6, 182, 212, 0.2) 0%, transparent 50%),
                linear-gradient(90deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              
              // Data Processing
              'tokenizer': `
                radial-gradient(circle at 25% 35%, rgba(245, 158, 11, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 75% 65%, rgba(251, 146, 60, 0.2) 0%, transparent 50%),
                linear-gradient(45deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'embeddings': `
                radial-gradient(circle at 35% 25%, rgba(244, 63, 94, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 65% 75%, rgba(236, 72, 153, 0.2) 0%, transparent 50%),
                linear-gradient(225deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              
              // Memory & Storage
              'memory-bank': `
                radial-gradient(circle at 20% 30%, rgba(16, 185, 129, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 80% 70%, rgba(6, 182, 212, 0.2) 0%, transparent 50%),
                linear-gradient(135deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'cache-layer': `
                radial-gradient(circle at 30% 20%, rgba(139, 92, 246, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 70% 80%, rgba(99, 102, 241, 0.2) 0%, transparent 50%),
                linear-gradient(315deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              
              // Optimization & Performance
              'optimizer': `
                radial-gradient(circle at 25% 25%, rgba(132, 204, 22, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 75% 75%, rgba(34, 197, 94, 0.2) 0%, transparent 50%),
                linear-gradient(180deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'scheduler': `
                radial-gradient(circle at 35% 30%, rgba(245, 158, 11, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 65% 70%, rgba(251, 191, 36, 0.2) 0%, transparent 50%),
                linear-gradient(90deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              
              // Security & Monitoring
              'security': `
                radial-gradient(circle at 30% 25%, rgba(239, 68, 68, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 70% 75%, rgba(244, 63, 94, 0.2) 0%, transparent 50%),
                linear-gradient(45deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'monitor': `
                radial-gradient(circle at 25% 35%, rgba(14, 165, 233, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 75% 65%, rgba(59, 130, 246, 0.2) 0%, transparent 50%),
                linear-gradient(225deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              
              // Advanced Features
              'attention': `
                radial-gradient(circle at 40% 30%, rgba(217, 70, 239, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 60% 70%, rgba(147, 51, 234, 0.2) 0%, transparent 50%),
                linear-gradient(135deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'inference': `
                radial-gradient(circle at 30% 40%, rgba(6, 182, 212, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 70% 20%, rgba(59, 130, 246, 0.2) 0%, transparent 50%),
                linear-gradient(315deg, #1f2937 0%, #374151 50%, #111827 100%)
              `,
              'fine-tune': `
                radial-gradient(circle at 35% 25%, rgba(34, 197, 94, 0.3) 0%, transparent 60%),
                radial-gradient(circle at 65% 75%, rgba(16, 185, 129, 0.2) 0%, transparent 50%),
                linear-gradient(180deg, #1f2937 0%, #374151 50%, #111827 100%)
              `
            };
            
            return (
              <div 
                key={node.id}
                className="absolute transform -translate-x-1/2 -translate-y-1/2" 
                style={{ 
                  left: node.position.x, 
                  top: node.position.y, 
                  zIndex: 10 
                }}
              >
                <div className="relative group">
                  {/* Hover Tooltip */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none z-50">
                    <div className="bg-black/90 border border-white/20 rounded-lg p-3 min-w-48 backdrop-blur-sm">
                      <div className="text-white text-sm font-medium mb-1">{node.label} {node.sublabel}</div>
                      <div className="text-gray-300 text-xs mb-2">Test ID: {node.id}</div>
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Accuracy:</span>
                          <span className={`${node.percentage > 95 ? 'text-green-400' : node.percentage > 85 ? 'text-yellow-400' : 'text-orange-400'}`}>
                            {Math.round(node.percentage)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Status:</span>
                          <span className={`${node.percentage > 95 ? 'text-green-400' : node.percentage > 85 ? 'text-yellow-400' : 'text-orange-400'}`}>
                            {node.percentage > 95 ? 'Excellent' : node.percentage > 85 ? 'Good' : 'Needs Improvement'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Test Type:</span>
                          <span className="text-blue-400">{node.label.includes('Training') ? 'Training Set' : 'Processing Node'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Samples:</span>
                          <span className="text-white">{Math.floor(Math.random() * 500) + 100}</span>
                        </div>
                      </div>
                      {/* Tooltip arrow */}
                      <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-white/20"></div>
                    </div>
                  </div>

                  {/* Subtle orbital ring */}
                  <div 
                    className="absolute rounded-full border border-white/10 animate-spin"
                    style={{ 
                      width: `${nodeSize + 12}px`, 
                      height: `${nodeSize + 12}px`,
                      left: '-6px',
                      top: '-6px',
                      animationDuration: `${20 + index * 4}s`
                    }}
                  >
                    <div className="absolute top-0 left-1/2 w-0.5 h-0.5 bg-white/40 rounded-full transform -translate-x-1/2 -translate-y-1/2 animate-pulse"></div>
                    <div className="absolute bottom-0 left-1/2 w-0.5 h-0.5 bg-gray-400/40 rounded-full transform -translate-x-1/2 translate-y-1/2 animate-pulse"></div>
                  </div>
                  
                  {/* Main cosmic sphere */}
                  <div 
                    className="relative rounded-full overflow-hidden shadow-lg border border-white/20 hover:scale-105 transition-all duration-500 flex items-center justify-center"
                    style={{ 
                      width: `${nodeSize}px`, 
                      height: `${nodeSize}px`,
                      transform: `scale(${0.8 + (node.percentage / 100) * 0.4})`,
                      background: spaceBackgrounds[node.id as keyof typeof spaceBackgrounds]
                    }}
                  >
                    {/* Subtle starfield */}
                    <div className="absolute inset-0">
                      {Array.from({ length: Math.floor(nodeSize / 16) }, (_, i) => (
                        <div
                          key={i}
                          className="absolute bg-white/30 rounded-full animate-pulse"
                          style={{
                            width: `${Math.random() * 1 + 0.3}px`,
                            height: `${Math.random() * 1 + 0.3}px`,
                            top: `${Math.random() * 100}%`,
                            left: `${Math.random() * 100}%`,
                            animationDelay: `${Math.random() * 6}s`,
                            animationDuration: `${3 + Math.random() * 3}s`
                          }}
                        />
                      ))}
                    </div>
                    
                    {/* Subtle nebula clouds */}
                    <div className="absolute inset-0 rounded-full opacity-20">
                      <div className="absolute inset-0 rounded-full animate-spin" style={{ animationDuration: `${30 + index * 3}s` }}>
                        <div 
                          className="absolute w-2 h-2 rounded-full blur-sm opacity-30"
                          style={{
                            background: `radial-gradient(circle, rgba(156, 163, 175, 0.3) 0%, transparent 70%)`,
                            top: '20%',
                            left: '30%'
                          }}
                        ></div>
                        <div 
                          className="absolute w-1 h-1 rounded-full blur-sm opacity-20"
                          style={{
                            background: `radial-gradient(circle, rgba(156, 163, 175, 0.2) 0%, transparent 70%)`,
                            bottom: '25%',
                            right: '25%'
                          }}
                        ></div>
                      </div>
                    </div>
                    
                    {/* Subtle energy waves */}
                    <div className="absolute inset-0 rounded-full">
                      <div 
                        className="absolute inset-0 rounded-full animate-pulse"
                        style={{
                          background: `conic-gradient(from 0deg, transparent, rgba(156, 163, 175, 0.1), transparent)`,
                          animationDuration: '5s'
                        }}
                      ></div>
                    </div>
                    
                    {/* Content - Only showing accuracy */}
                    <div className="relative z-10 text-center">
                      <div 
                        className="text-white font-bold tracking-wide drop-shadow-lg"
                        style={{ 
                          fontSize: `${Math.max(12, size / 2.5)}px`,
                          textShadow: `0 0 8px ${node.color.split(' ')[1]}, 0 0 16px ${node.color.split(' ')[1]}/50`
                        }}
                      >
                        {Math.round(node.percentage)}%
                      </div>
                    </div>
                    
                    {/* Atmospheric glow */}
                    <div className="absolute inset-0 rounded-full bg-gradient-radial from-white/5 via-transparent to-transparent animate-pulse"></div>
                  </div>
                  
                  {/* Outer space glow */}
                  <div 
                    className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-lg"
                    style={{
                      background: `radial-gradient(circle, ${node.color.split(' ')[1]}/30 0%, transparent 70%)`,
                      width: `${nodeSize + 20}px`,
                      height: `${nodeSize + 20}px`,
                      left: '-10px',
                      top: '-10px'
                    }}
                  ></div>
                  
                  {/* High performance cosmic ring */}
                  {node.percentage > 95 && (
                    <div 
                      className="absolute rounded-full border-2 animate-ping"
                      style={{ 
                        width: `${nodeSize + 12}px`, 
                        height: `${nodeSize + 12}px`,
                        left: '-6px',
                        top: '-6px',
                        borderColor: `${node.color.split(' ')[1]}`,
                        boxShadow: `0 0 20px ${node.color.split(' ')[1]}/50`
                      }}
                    ></div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        </div>
      </div>

      {/* Processing Type Selection Modal */}
      {showProcessingModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-gray-900 border border-gray-600 rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-white text-xl font-light mb-4">Select Processing Method</h3>
            
             {/* GPU Detection Status */}
             <div className="mb-6">
               <div className="flex items-center space-x-2 mb-2">
                 <div className={`w-3 h-3 rounded-full ${(hasNvidiaGpu || manualGpuOverride) ? 'bg-green-500' : 'bg-red-500'}`}></div>
                 <span className="text-gray-300 text-sm">
                   NVIDIA GPU: {hasNvidiaGpu === null ? 'Detecting...' : hasNvidiaGpu ? 'Detected' : manualGpuOverride ? 'Manual Override' : 'Not Found'}
                 </span>
               </div>
               {(hasNvidiaGpu || manualGpuOverride) && (
                 <p className="text-green-400 text-xs">✓ Local GPU processing available</p>
               )}
               {!hasNvidiaGpu && hasNvidiaGpu !== null && !manualGpuOverride && (
                 <div className="mt-2">
                   <p className="text-yellow-400 text-xs mb-2">⚠ GPU not detected automatically</p>
                   <label className="flex items-center space-x-2 cursor-pointer">
                     <input
                       type="checkbox"
                       checked={manualGpuOverride}
                       onChange={(e) => setManualGpuOverride(e.target.checked)}
                       className="text-blue-500"
                     />
                     <span className="text-gray-300 text-xs">I have an NVIDIA GPU (manual override)</span>
                   </label>
                   <button
                     onClick={async () => {
                       console.log('🔍 Running GPU detection debug...');
                       const detected = await detectNvidiaGpu();
                       setHasNvidiaGpu(detected);
                       alert(`GPU Detection Results:\nCheck the browser console (F12) for detailed information.\nDetected: ${detected ? 'Yes' : 'No'}`);
                     }}
                     className="mt-2 px-3 py-1 text-xs bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
                   >
                     🔍 Debug GPU Detection
                   </button>
                 </div>
               )}
             </div>

            {/* Processing Options */}
            <div className="space-y-3 mb-6">
              {/* Cloud Processing Option */}
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="processingType"
                  value="cloud"
                  checked={processingType === 'cloud'}
                  onChange={(e) => setProcessingType(e.target.value as 'cloud' | 'gpu')}
                  className="text-blue-500"
                />
                <div>
                  <div className="text-white font-medium">Cloud Processing</div>
                  <div className="text-gray-400 text-sm">Use remote cloud services (default)</div>
                </div>
              </label>

               {/* GPU Processing Option */}
               <label className={`flex items-center space-x-3 cursor-pointer ${!(hasNvidiaGpu || manualGpuOverride) ? 'opacity-50' : ''}`}>
                 <input
                   type="radio"
                   name="processingType"
                   value="gpu"
                   checked={processingType === 'gpu'}
                   onChange={(e) => setProcessingType(e.target.value as 'cloud' | 'gpu')}
                   disabled={!(hasNvidiaGpu || manualGpuOverride)}
                   className="text-green-500"
                 />
                 <div>
                   <div className="text-white font-medium">Local GPU Processing</div>
                   <div className="text-gray-400 text-sm">
                     {(hasNvidiaGpu || manualGpuOverride) ? 'Use your NVIDIA GPU (faster, private)' : 'Requires NVIDIA GPU'}
                   </div>
                 </div>
               </label>
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-3">
              <button
                onClick={() => setShowProcessingModal(false)}
                className="flex-1 px-4 py-2 border border-gray-600 text-gray-300 rounded hover:bg-gray-800 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleProcessingStart}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              >
                Start {processingType === 'gpu' ? 'GPU' : 'Cloud'} Processing
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
