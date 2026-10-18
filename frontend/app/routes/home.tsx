import type { Route } from "./+types/home";

import { ChatPage } from "~/components/chat-page";
import { ChatProvider } from "~/providers/chat-provider";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Rwandan legal assistant" },
    { name: "description", content: "Ask a question about Rwandan law." },
  ];
}

export default function Home() {
  return (
    <ChatProvider>
      <ChatPage />
    </ChatProvider>
  );
}
