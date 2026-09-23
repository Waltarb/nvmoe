import type { User } from "./types";

export function formatUser(user: User): string {
  return user.fullName;
}

export function initials(user: User): string {
  return user.fullName
    .split(" ")
    .map((p) => p[0])
    .join("")
    .toUpperCase();
}
