import type { User } from "./types";

export interface RawUser {
  id: string;
  name: string;
}

export function toUser(raw: RawUser): User {
  return { id: raw.id, fullName: raw.name.trim() };
}
