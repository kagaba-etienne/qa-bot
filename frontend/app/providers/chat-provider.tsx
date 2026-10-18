import { createContext, useContext, useState, type ReactNode } from "react";

import { useSendChatMessage } from "~/hooks/use-chat";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type ChatContextValue = {
  messages: ChatMessage[];
  draft: string;
  setDraft: (value: string) => void;
  send: () => void;
  isPending: boolean;
  error: string | null;
};

const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const mutation = useSendChatMessage();

  const send = () => {
    const message = draft.trim();
    if (!message || mutation.isPending) {
      return;
    }
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: message },
    ]);
    setDraft("");
    mutation.mutate(message, {
      onSuccess: (reply: string) => {
        setMessages((current) => [
          ...current,
          { id: crypto.randomUUID(), role: "assistant", content: reply },
        ]);
      },
    });
  };

  return (
    <ChatContext.Provider
      value={{
        messages,
        draft,
        setDraft,
        send,
        isPending: mutation.isPending,
        error: mutation.isError
          ? "The legal assistant could not answer right now."
          : null,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const value = useContext(ChatContext);
  if (!value) {
    throw new Error("useChat must be used within ChatProvider.");
  }
  return value;
}
