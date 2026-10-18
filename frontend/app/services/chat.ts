type ChatReply = {
  reply: string;
};

export async function sendChatMessage(message: string): Promise<string> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!response.ok) {
    throw new Error("The legal assistant could not answer right now.");
  }
  const data = (await response.json()) as ChatReply;
  if (typeof data.reply !== "string") {
    throw new Error("The legal assistant could not answer right now.");
  }
  return data.reply;
}
