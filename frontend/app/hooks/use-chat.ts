import { useMutation } from "@tanstack/react-query";

import { sendChatMessage } from "~/services/chat";

export function useSendChatMessage() {
  return useMutation({
    mutationFn: sendChatMessage,
  });
}
