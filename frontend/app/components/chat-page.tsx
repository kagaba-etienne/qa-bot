import { useEffect, useRef } from "react";

import { useChat } from "~/providers/chat-provider";

export function ChatPage() {
  const { messages, draft, setDraft, send, isPending, error } = useChat();
  const transcriptRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const transcript = transcriptRef.current;
    if (transcript) {
      transcript.scrollTop = transcript.scrollHeight;
    }
  }, [messages, isPending]);

  return (
    <main className="mx-auto flex h-dvh w-full max-w-3xl flex-col">
      <header className="border-b border-gray-200 px-4 py-4 dark:border-gray-800">
        <h1 className="text-lg font-semibold">Rwandan legal assistant</h1>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Ask a question about Rwandan law. Answers use the legal text retrieved
          for your question.
        </p>
      </header>
      <div ref={transcriptRef} className="flex-1 space-y-4 overflow-y-auto px-4 py-6">
        {messages.length === 0 && !isPending ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Send a question to start.
          </p>
        ) : null}
        {messages.map((message) => (
          <article
            key={message.id}
            className={
              message.role === "user"
                ? "ml-auto max-w-[85%] rounded-2xl bg-gray-900 px-4 py-3 text-white dark:bg-gray-100 dark:text-gray-950"
                : "max-w-[85%] rounded-2xl bg-gray-100 px-4 py-3 text-gray-950 dark:bg-gray-900 dark:text-gray-100"
            }
          >
            <p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>
          </article>
        ))}
        {isPending ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">Thinking...</p>
        ) : null}
      </div>
      <form
        className="border-t border-gray-200 px-4 py-4 dark:border-gray-800"
        onSubmit={(event) => {
          event.preventDefault();
          send();
        }}
      >
        {error ? (
          <p className="mb-3 text-sm text-red-700 dark:text-red-400" role="alert">
            {error}
          </p>
        ) : null}
        <div className="flex gap-2">
          <label className="sr-only" htmlFor="chat-message">
            Message
          </label>
          <textarea
            id="chat-message"
            className="min-h-12 flex-1 resize-none rounded-xl border border-gray-300 bg-white px-3 py-2 text-sm outline-none focus:border-gray-900 dark:border-gray-700 dark:bg-gray-950 dark:focus:border-gray-300"
            rows={2}
            value={draft}
            placeholder="Ask about Rwandan law"
            disabled={isPending}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                send();
              }
            }}
          />
          <button
            type="submit"
            className="rounded-xl bg-gray-900 px-4 text-sm font-medium text-white disabled:opacity-50 dark:bg-gray-100 dark:text-gray-950"
            disabled={isPending || draft.trim().length === 0}
          >
            Send
          </button>
        </div>
      </form>
    </main>
  );
}
