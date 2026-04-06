import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import OpenAI from 'openai';
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { randomUUID } from 'crypto';

dotenv.config();

const app = express();
const __dirname = path.dirname(fileURLToPath(import.meta.url));

app.use(express.json({ limit: '10mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Supabase ──
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// ── AI clients ──
const anthropic = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
  : null;

const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

// xAI uses an OpenAI-compatible API
const xai = process.env.XAI_API_KEY
  ? new OpenAI({ apiKey: process.env.XAI_API_KEY, baseURL: 'https://api.x.ai/v1' })
  : null;

const MODELS = {
  anthropic: [
    { id: 'claude-opus-4-6',           name: 'Claude Opus 4.6',   description: 'Most capable',       contextWindow: 200000 },
    { id: 'claude-sonnet-4-6',         name: 'Claude Sonnet 4.6', description: 'Balanced',           contextWindow: 200000 },
    { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5',  description: 'Fastest',            contextWindow: 200000 },
  ],
  openai: [
    { id: 'gpt-4.1',      name: 'GPT-4.1',       description: 'Latest flagship',     contextWindow: 1047576 },
    { id: 'gpt-4.1-mini', name: 'GPT-4.1 mini',  description: 'Fast & efficient',    contextWindow: 1047576 },
    { id: 'gpt-4.1-nano', name: 'GPT-4.1 nano',  description: 'Lightest & cheapest', contextWindow: 1047576 },
    { id: 'o3',           name: 'o3',             description: 'Advanced reasoning',  contextWindow: 200000  },
    { id: 'o4-mini',      name: 'o4-mini',        description: 'Fast reasoning',      contextWindow: 200000  },
  ],
  xai: [
    { id: 'grok-4',                       name: 'Grok 4',                  description: 'Latest flagship',    contextWindow: 256000 },
    { id: 'grok-4.20-0309-reasoning',     name: 'Grok 4.20 Reasoning',     description: 'Extended reasoning', contextWindow: 256000 },
    { id: 'grok-4.20-0309-non-reasoning', name: 'Grok 4.20 Standard',      description: 'Fast, no reasoning', contextWindow: 256000 },
    { id: 'grok-4-1-fast-reasoning',      name: 'Grok 4.1 Fast Reasoning', description: 'Speed + reasoning',  contextWindow: 256000 },
    { id: 'grok-4-1-fast-non-reasoning',  name: 'Grok 4.1 Fast',           description: 'Fastest & cheapest', contextWindow: 256000 },
    { id: 'grok-3',                       name: 'Grok 3',                  description: 'Previous generation',contextWindow: 131072 },
    { id: 'grok-3-mini',                  name: 'Grok 3 Mini',             description: 'Previous, lightweight',contextWindow: 131072 },
  ],
};

// ── Chat CRUD routes ──
app.get('/api/models', (req, res) => res.json(MODELS));

app.get('/api/chats', async (req, res) => {
  const { data, error } = await supabase
    .from('chats')
    .select('id, title, provider, model, created_at, updated_at')
    .order('updated_at', { ascending: false });
  if (error) { console.error('GET /api/chats error:', error); return res.status(500).json({ error: error.message }); }
  res.json(data);
});

app.post('/api/chats', async (req, res) => {
  const { title, provider, model } = req.body;
  if (!title || !provider || !model) return res.status(400).json({ error: 'Missing fields' });
  const now = Date.now();
  const { data, error } = await supabase
    .from('chats')
    .insert({ id: randomUUID(), title, provider, model, created_at: now, updated_at: now })
    .select('id')
    .single();
  if (error) return res.status(500).json({ error: error.message });
  res.json({ id: data.id });
});

app.get('/api/chats/:id', async (req, res) => {
  const [{ data: chat, error: e1 }, { data: msgs, error: e2 }] = await Promise.all([
    supabase.from('chats').select('*').eq('id', req.params.id).single(),
    supabase.from('messages').select('role, content').eq('chat_id', req.params.id).order('id'),
  ]);
  if (e1) return res.status(404).json({ error: 'Not found' });
  if (e2) return res.status(500).json({ error: e2.message });
  res.json({ ...chat, messages: msgs });
});

app.delete('/api/chats/:id', async (req, res) => {
  const { error } = await supabase.from('chats').delete().eq('id', req.params.id);
  if (error) return res.status(500).json({ error: error.message });
  res.json({ ok: true });
});

app.patch('/api/chats/:id/title', async (req, res) => {
  const { title } = req.body;
  if (!title) return res.status(400).json({ error: 'Missing title' });
  const { error } = await supabase
    .from('chats')
    .update({ title, updated_at: Date.now() })
    .eq('id', req.params.id);
  if (error) return res.status(500).json({ error: error.message });
  res.json({ ok: true });
});

app.post('/api/chats/:id/generate-title', async (req, res) => {
  const { provider, model, firstUserMessage } = req.body;
  if (!firstUserMessage) return res.status(400).json({ error: 'Missing firstUserMessage' });

  const prompt = `Summarize the topic of this message as a short chat title (4-6 words max). Reply with just the title, no quotes, no period:\n\n"${firstUserMessage.slice(0, 300)}"`;

  // Use a fast/cheap model from the same provider
  const fastModels = {
    anthropic: 'claude-haiku-4-5-20251001',
    openai:    'gpt-4.1-nano',
    xai:       'grok-3-mini',
  };

  let title = null;
  try {
    if (provider === 'anthropic' && anthropic) {
      const msg = await anthropic.messages.create({
        model: fastModels.anthropic,
        max_tokens: 20,
        messages: [{ role: 'user', content: prompt }],
      });
      title = msg.content[0]?.text?.trim();
    } else if (provider === 'openai' && openai) {
      const completion = await openai.chat.completions.create({
        model: fastModels.openai,
        max_tokens: 20,
        messages: [{ role: 'user', content: prompt }],
      });
      title = completion.choices[0]?.message?.content?.trim();
    } else if (provider === 'xai' && xai) {
      const completion = await xai.chat.completions.create({
        model: fastModels.xai,
        max_tokens: 20,
        messages: [{ role: 'user', content: prompt }],
      });
      title = completion.choices[0]?.message?.content?.trim();
    }

    if (title) {
      await supabase.from('chats').update({ title, updated_at: Date.now() }).eq('id', req.params.id);
    }
    res.json({ title });
  } catch (err) {
    console.error('Title generation error:', err.message);
    res.json({ title: null }); // fail silently — original title stays
  }
});

app.post('/api/chats/:id/messages', async (req, res) => {
  const { role, content } = req.body;
  if (!role || !content) return res.status(400).json({ error: 'Missing fields' });
  const now = Date.now();
  const [{ error: e1 }, { error: e2 }] = await Promise.all([
    supabase.from('messages').insert({ chat_id: req.params.id, role, content, created_at: now }),
    supabase.from('chats').update({ updated_at: now }).eq('id', req.params.id),
  ]);
  if (e1) return res.status(500).json({ error: e1.message });
  res.json({ ok: true });
});

// ── Streaming chat ──
app.post('/api/chat', async (req, res) => {
  const { messages, model, provider } = req.body;

  if (!model || !provider || !messages?.length) {
    return res.status(400).json({ error: 'Missing required fields: messages, model, provider' });
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

  try {
    if (provider === 'anthropic') {
      if (!anthropic) return send({ error: 'ANTHROPIC_API_KEY not configured' });

      const stream = anthropic.messages.stream({ model, max_tokens: 8192, messages });
      for await (const chunk of stream) {
        if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
          send({ text: chunk.delta.text });
        }
      }
    } else if (provider === 'openai') {
      if (!openai) return send({ error: 'OPENAI_API_KEY not configured' });

      const isReasoningModel = model.startsWith('o3') || model.startsWith('o4');
      if (isReasoningModel) {
        const filteredMessages = messages.filter((m) => m.role !== 'system');
        const completion = await openai.chat.completions.create({ model, messages: filteredMessages });
        const text = completion.choices[0]?.message?.content || '';
        const words = text.split(' ');
        for (let i = 0; i < words.length; i++) {
          send({ text: (i === 0 ? '' : ' ') + words[i] });
          await new Promise((r) => setTimeout(r, 5));
        }
      } else {
        const stream = await openai.chat.completions.create({ model, messages, stream: true });
        for await (const chunk of stream) {
          const text = chunk.choices[0]?.delta?.content;
          if (text) send({ text });
        }
      }
    } else if (provider === 'xai') {
      if (!xai) return send({ error: 'XAI_API_KEY not configured' });

      const stream = await xai.chat.completions.create({ model, messages, stream: true });
      for await (const chunk of stream) {
        const text = chunk.choices[0]?.delta?.content;
        if (text) send({ text });
      }
    } else {
      send({ error: `Unknown provider: ${provider}` });
    }
  } catch (err) {
    console.error('Chat error:', err);
    send({ error: err.message || 'An error occurred' });
  }

  res.write('data: [DONE]\n\n');
  res.end();
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n  Chat UI running at http://localhost:${PORT}\n`);
  if (!process.env.ANTHROPIC_API_KEY) console.warn('  Warning: ANTHROPIC_API_KEY not set');
  if (!process.env.OPENAI_API_KEY)    console.warn('  Warning: OPENAI_API_KEY not set');
  if (!process.env.XAI_API_KEY)       console.warn('  Warning: XAI_API_KEY not set');
});
