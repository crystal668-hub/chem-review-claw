import { existsSync, lstatSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";

const EXEC_TOOL_NAMES = new Set(["exec", "execute", "shell", "bash", "command"]);
const ABSOLUTE_PATH_RE = /(?<![A-Za-z0-9_$])\/(?:[^\s"'`;&|()<>\n\\]+)/g;
const RELATIVE_ESCAPE_RE = /(?<![A-Za-z0-9_$])(?:\.\/)*\.\.\/(?:[^\s"'`;&|()<>\n\\]+)/g;
const HOME_PATH_RE = /(?<![A-Za-z0-9_$])~(?:\/[^\s"'`;&|()<>\n\\]*)?/;
const FORBIDDEN_DISTRIBUTIONS = new Set(["verifier-grounded-benchmark"]);
const SYSTEM_PATH_PREFIXES = [
  "/bin/",
  "/sbin/",
  "/usr/bin/",
  "/usr/sbin/",
  "/usr/local/bin/",
  "/opt/homebrew/bin/",
];
const TOOL_RULES = {
  read: [{ keys: ["path", "file_path"], access: "read" }],
  read_file: [{ keys: ["path", "file_path"], access: "read" }],
  image: [{ keys: ["path", "file_path"], access: "read" }],
  open_file: [{ keys: ["path", "file_path"], access: "read" }],
  list: [{ keys: ["path", "directory"], access: "list" }],
  list_directory: [{ keys: ["path", "directory"], access: "list" }],
  search: [{ keys: ["path", "directory"], access: "search" }],
  find: [{ keys: ["path", "directory"], access: "search" }],
  glob: [{ keys: ["path", "directory"], access: "search" }],
  write: [{ keys: ["path", "file_path", "target"], access: "write" }],
  write_file: [{ keys: ["path", "file_path", "target"], access: "write" }],
  create: [{ keys: ["path", "file_path", "target"], access: "write" }],
  create_file: [{ keys: ["path", "file_path", "target"], access: "write" }],
  edit: [{ keys: ["path", "file_path", "target"], access: "mutate" }],
  delete: [{ keys: ["path", "file_path", "target"], access: "mutate" }],
  remove: [{ keys: ["path", "file_path", "target"], access: "mutate" }],
  move: [
    { keys: ["source"], access: "read" },
    { keys: ["target"], access: "mutate" },
  ],
  rename: [
    { keys: ["source"], access: "read" },
    { keys: ["target"], access: "mutate" },
  ],
  copy: [
    { keys: ["source"], access: "read" },
    { keys: ["target"], access: "mutate" },
  ],
};
const PATH_KEYS = new Set(["path", "file_path", "directory", "workdir", "cwd", "source", "target"]);

function isContained(root, candidate) {
  const rel = relative(root, candidate);
  return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

function resolveWithExistingPrefix(candidate) {
  let current = candidate;
  const suffix = [];
  while (!existsSync(current)) {
    const parent = dirname(current);
    if (parent === current) break;
    suffix.unshift(current.slice(parent.length + (parent.endsWith(sep) ? 0 : 1)));
    current = parent;
  }
  const resolvedPrefix = existsSync(current) ? realpathSync(current) : current;
  return suffix.reduce((path, part) => resolve(path, part), resolvedPrefix);
}

function workspaceRoot(policy) {
  const scope = (policy?.read_scopes || []).find((item) => item?.scope_id === "active_workspace");
  return typeof scope?.path === "string" ? realpathSync(scope.path) : null;
}

function scopesFor(policy, access) {
  if (access === "write" || access === "mutate") return policy?.write_scopes || [];
  if (access === "workdir") return policy?.exec_workdir_scopes || [];
  return policy?.read_scopes || [];
}

function protectedRoots(policy) {
  return Array.isArray(policy?.protected_roots)
    ? policy.protected_roots
        .map((item) => {
          if (typeof item?.path !== "string") return null;
          try {
            return resolveWithExistingPrefix(resolve(item.path));
          } catch {
            return null;
          }
        })
        .filter(Boolean)
    : [];
}

function scopeAllows(scope, candidate) {
  const configured = resolveWithExistingPrefix(scope.path);
  return scope.kind === "file" ? candidate === configured : isContained(configured, candidate);
}

function validateCandidate({ policy, rawPath, access }) {
  if (typeof rawPath !== "string" || rawPath.trim() === "") return { ok: true };
  let root;
  try {
    root = workspaceRoot(policy);
  } catch {
    return { ok: false, reason: "configured benchmark workspace is unavailable", candidate: rawPath };
  }
  if (!root) return { ok: false, reason: "active workspace scope is missing", candidate: rawPath };
  const requested = rawPath.trim();
  const lexical = isAbsolute(requested) ? requested : resolve(root, requested);
  let candidate;
  try {
    candidate = resolveWithExistingPrefix(lexical);
  } catch {
    return { ok: false, reason: "path cannot be resolved safely", candidate: lexical };
  }
  const scopes = scopesFor(policy, access);
  if (!scopes.some((scope) => scopeAllows(scope, candidate))) {
    return { ok: false, reason: `${access} path is outside the policy scope`, candidate };
  }
  if (access === "workdir") {
    try {
      if (!lstatSync(candidate).isDirectory()) {
        return { ok: false, reason: "exec.workdir is not a directory", candidate };
      }
    } catch {
      return { ok: false, reason: "exec.workdir does not exist", candidate };
    }
  }
  return { ok: true, candidate };
}

function trimPathPunctuation(value) {
  return value.replace(/[.,;:!?\]})>]+$/g, "");
}

function isSystemPath(candidate) {
  return SYSTEM_PATH_PREFIXES.some((prefix) => candidate.startsWith(prefix)) || candidate.startsWith("/dev/");
}

function validateExecCommand({ policy, command }) {
  if (typeof command !== "string" || command.trim() === "") return { ok: true };

  const workspace = workspaceRoot(policy);
  const roots = protectedRoots(policy);
  if (HOME_PATH_RE.test(command)) {
    return {
      ok: false,
      access: "exec",
      reason: "exec command references a home-relative path outside the policy scope",
      candidate: "~",
    };
  }
  const candidates = new Set();
  for (const match of command.matchAll(ABSOLUTE_PATH_RE)) {
    const raw = trimPathPunctuation(match[0]);
    if (!raw) continue;
    try {
      candidates.add(resolveWithExistingPrefix(raw));
    } catch {
      return { ok: false, access: "exec", reason: "exec command contains a path that cannot be resolved safely", candidate: raw };
    }
  }

  for (const candidate of candidates) {
    const protectedRoot = roots.find((root) => isContained(root, candidate));
    if (protectedRoot) {
      return {
        ok: false,
        access: "exec",
        reason: "exec command references a protected path",
        candidate,
      };
    }
    if (!isSystemPath(candidate) && !isContained(workspace, candidate)) {
      return {
        ok: false,
        access: "exec",
        reason: "exec command references a path outside the policy scope",
        candidate,
      };
    }
  }
  for (const match of command.matchAll(RELATIVE_ESCAPE_RE)) {
    const raw = trimPathPunctuation(match[0]);
    try {
      candidates.add(resolveWithExistingPrefix(resolve(workspace, raw)));
    } catch {
      return { ok: false, access: "exec", reason: "exec command contains a path that cannot be resolved safely", candidate: raw };
    }
  }
  return { ok: true };
}

function normalizeDistribution(value) {
  return String(value || "").toLowerCase().replace(/[_.]+/g, "-").split(/[<>=!~\[]/, 1)[0];
}

function validateAttemptPackageCommand(command) {
  if (!process.env.BENCHMARK_ATTEMPT_PYTHON || typeof command !== "string") return { ok: true };
  const tokens = command.trim().split(/\s+/);
  const joined = ` ${tokens.join(" ")} `;
  if (/(?:^|[;&|]\s*)(?:python\s+-m\s+pip|pip3?|\S+\/pip3?)\s+(?:install|uninstall|download|wheel)\b/i.test(command.trim())) {
    return { ok: false, access: "dependency", reason: "pip mutations are disabled; use uv pip with the attempt environment" };
  }
  if (/\b(?:git\+|https?:\/\/|file:)/i.test(command) || /(?:^|\s)(?:-e|--editable|-f|--find-links|--index|--index-url|--extra-index-url|--no-index)(?:\s|=)/i.test(command)) {
    return { ok: false, access: "dependency", reason: "only packages from the configured PyPI registry are allowed" };
  }
  if (/\buv\s+(?:sync|add|remove|tool|venv)\b/i.test(command) || /\buv\s+run\b[^;&|]*(?:--with|--with-requirements)/i.test(command)) {
    return { ok: false, access: "dependency", reason: "dependency changes must use uv pip with the attempt environment" };
  }
  if (/\buv\s+pip\s+(?:install|uninstall)\b[^;&|]*(?:--python|--system|--target|--prefix|--project)(?:\s|=)/i.test(command)) {
    return { ok: false, access: "dependency", reason: "the attempt dependency target cannot be overridden" };
  }
  const marker = tokens.findIndex((value, index) => value === "pip" && index > 0 && tokens[index - 1] === "uv");
  if (marker >= 0 && ["install", "uninstall"].includes(tokens[marker + 1])) {
    for (const token of tokens.slice(marker + 2)) {
      if (token.startsWith("-") || token.includes("/")) continue;
      if (FORBIDDEN_DISTRIBUTIONS.has(normalizeDistribution(token))) {
        return { ok: false, access: "dependency", reason: "the requested distribution is forbidden in benchmark attempts" };
      }
    }
  }
  if (joined.includes(" UV_EXCLUDE_NEWER=") || joined.includes(" UV_DEFAULT_INDEX=") || joined.includes(" UV_CACHE_DIR=")) {
    return { ok: false, access: "dependency", reason: "attempt dependency policy variables cannot be overridden" };
  }
  return { ok: true };
}

export function validateToolCall({ policy, toolName, params = {} }) {
  const normalizedTool = String(toolName || "").toLowerCase();
  const checks = [];
  if (EXEC_TOOL_NAMES.has(normalizedTool)) {
    const packageValidation = validateAttemptPackageCommand(params.command);
    if (!packageValidation.ok) return packageValidation;
    const commandValidation = validateExecCommand({ policy, command: params.command });
    if (!commandValidation.ok) return commandValidation;
    for (const key of ["workdir", "cwd"]) {
      if (key in params) checks.push({ key, rawPath: params[key], access: "workdir" });
    }
  } else if (TOOL_RULES[normalizedTool]) {
    for (const rule of TOOL_RULES[normalizedTool]) {
      for (const key of rule.keys) {
        if (key in params) checks.push({ key, rawPath: params[key], access: rule.access });
      }
    }
  } else {
    for (const [key, value] of Object.entries(params || {})) {
      if (PATH_KEYS.has(String(key).toLowerCase())) {
        checks.push({ key, rawPath: value, access: "unknown" });
      }
    }
  }
  for (const check of checks) {
    const validation = validateCandidate({ policy, rawPath: check.rawPath, access: check.access });
    if (!validation.ok) return { ...validation, ...check };
  }
  return { ok: true };
}

export function validateExecWorkdir({ workspaceRoot: root, workdir }) {
  const resolved = resolve(root);
  const policy = {
    read_scopes: [{ scope_id: "active_workspace", path: resolved, kind: "directory" }],
    write_scopes: [{ scope_id: "attempt_scratch", path: resolve(resolved, "scratch"), kind: "directory" }],
    exec_workdir_scopes: [{ scope_id: "active_workspace", path: resolved, kind: "directory" }],
  };
  return validateToolCall({ policy, toolName: "exec", params: { workdir } });
}

export default {
  id: "benchmark-workdir-guard",
  name: "Benchmark Workspace Guard",
  description: "Blocks structured filesystem operations outside the current benchmark policy.",
  register(api) {
    const configured = api.pluginConfig?.agentPolicies;
    const agentPolicies = configured && typeof configured === "object" ? configured : {};
    api.on(
      "before_tool_call",
      (event, context) => {
        const agentId = String(context.agentId || "");
        const policy = agentPolicies[agentId];
        if (!policy || typeof policy !== "object") return;
        const validation = validateToolCall({
          policy,
          toolName: event.toolName,
          params: event.params || {},
        });
        if (validation.ok) return;
        return {
          block: true,
          blockReason: [
            "benchmark_workspace_guard_blocked",
            `policy=${policy.policy_digest || "unknown"}`,
            `access=${validation.access || "unknown"}`,
            `candidate=${validation.candidate || "unknown"}`,
            `reason=${validation.reason}`,
            "The operation was not executed. Use a workspace-relative scratch/... path or the current scratch environment variable.",
          ].join(" "),
        };
      },
      { priority: 100 },
    );
  },
};
