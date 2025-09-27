# 🔧 Fix Your Supabase Credentials

## ❌ Problem Found: Invalid API Key

Your `.env.local` file has an **invalid Supabase API key**. The test shows:
```
❌ Chats table error: {
  message: 'Invalid API key',
  hint: 'Double check your Supabase `anon` or `service_role` API key.'
}
```

## 🛠️ How to Fix This

### Step 1: Get Correct Credentials from Supabase

1. **Go to your Supabase dashboard**: [app.supabase.com](https://app.supabase.com)
2. **Select your project** (the one with URL: `tjcwflakfbdghyniijun.supabase.co`)
3. **Go to Settings** → **API**
4. **Copy the correct values**:
   - **Project URL**: Should be `https://tjcwflakfbdghyniijun.supabase.co`
   - **anon public key**: Should start with `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.` and be much longer

### Step 2: Update Your .env.local File

Replace the contents of `/Users/kurtis/Desktop/shellhacks/shellhacks/next/.env.local` with:

```bash
NEXT_PUBLIC_SUPABASE_URL=https://tjcwflakfbdghyniijun.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.CORRECT_KEY_FROM_DASHBOARD
```

**⚠️ Important**: 
- Use the **anon public** key, NOT the **service_role** key
- The key should be much longer than what you currently have
- Copy it exactly from the Supabase dashboard

### Step 3: Create Database Tables

Once you have the correct API key, you need to create the database tables. Go to **SQL Editor** in your Supabase dashboard and run:

```sql
-- Create chats table
CREATE TABLE chats (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id TEXT
);

-- Create messages table
CREATE TABLE messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    is_user BOOLEAN NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_messages_chat_id ON messages(chat_id);
CREATE INDEX idx_chats_created_at ON chats(created_at DESC);
CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (for testing)
CREATE POLICY "Enable read access for all users" ON chats FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON chats FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON chats FOR UPDATE USING (true);

CREATE POLICY "Enable read access for all users" ON messages FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON messages FOR INSERT WITH CHECK (true);
```

### Step 4: Restart Your Development Server

After updating `.env.local`:

```bash
# Stop current server (Ctrl+C)
# Then restart:
npm run dev
```

### Step 5: Test the Connection

After restarting, check your browser console. You should see:

```
🔍 Supabase configuration debug: { configured: true, hasUrl: true, hasKey: true }
🔍 Testing Supabase connection...
✅ Supabase client configured, testing database connection...
🎉 Supabase connection successful!
📊 Database ready - will use Supabase for storage
```

## 🎯 Why This Happened

Your current API key looks truncated or malformed:
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9eyJpc3MiOiJzdXBhYmFzZSI...
```

A valid Supabase anon key should be much longer and have proper JWT structure with dots (`.`) separating the parts.

## ✅ After Fixing

Once you have the correct credentials and tables:

1. **Your chats will save to Supabase database** ✅
2. **You'll see data in Table Editor** ✅
3. **Console will show success messages** ✅
4. **Data will persist across devices** ✅

## 🆘 Still Need Help?

If you're having trouble finding the correct API key:
1. Screenshot your Supabase Settings → API page
2. Make sure you're copying the **anon public** key (not service_role)
3. The key should be 200+ characters long

Let me know once you've updated the credentials and I can help verify the connection! 🚀
