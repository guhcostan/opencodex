import { describe, expect, test } from "bun:test";
import { createResponsesPassthroughAdapter as createResponsesPassthroughAdapterProduction } from "../src/adapters/openai-responses";
import { getProviderRegistryEntry } from "../src/providers/registry";
import type { OcxProviderConfig } from "../src/types";
import { withTestTranslatorBudget } from "./helpers/translator-budget";

const createResponsesPassthroughAdapter = (...args: Parameters<typeof createResponsesPassthroughAdapterProduction>) =>
  withTestTranslatorBudget(createResponsesPassthroughAdapterProduction(...args));

const PROVIDER = {
  adapter: "openai-responses",
  baseUrl: "https://opencode.ai/zen/v1",
  apiKey: "test-key",
} as unknown as OcxProviderConfig;

/** A Codex web_search declaration exactly as `hosted_spec.rs` emits it for TextAndImage. */
function webSearchTool(): Record<string, unknown> {
  return {
    type: "web_search",
    search_content_types: ["text", "image"],
    search_context_size: "medium",
  };
}

function build(modelId: string, rawBody: Record<string, unknown>): Record<string, unknown> {
  const request = createResponsesPassthroughAdapter(PROVIDER).buildRequest({
    modelId,
    context: { messages: [] },
    stream: true,
    options: {},
    _rawBody: { model: modelId, input: "ping", ...rawBody },
  }, { headers: new Headers() });
  return JSON.parse(request.body) as Record<string, unknown>;
}

const toolsOf = (body: Record<string, unknown>) => body.tools as Array<Record<string, unknown>>;

/**
 * Muse Spark's Responses gateway 400s a plain `web_search` carrying
 * `search_content_types`, while accepting the same field on `web_search_preview` and
 * accepting a bare `web_search` (#2617).
 *
 * The field is not ours: Codex emits it from `web_search_tool_type: TextAndImage`. This is
 * the same incompatibility class Codex itself handles for Bedrock by selecting text-only
 * search, so dropping exactly the refused field at the adapter boundary is a compatibility
 * guard rather than a symptom patch — the tool type and every other accepted option survive.
 */
describe("#2617 Muse Spark web_search compatibility", () => {
  test("drops search_content_types from a plain web_search, keeping the tool and its other fields", () => {
    const body = build("muse-spark-1.2-contributor", { tools: [webSearchTool()] });
    const tool = toolsOf(body)[0]!;
    expect(tool.type).toBe("web_search");
    expect(tool.search_context_size).toBe("medium");
    expect(Object.hasOwn(tool, "search_content_types")).toBe(false);
  });

  test("web_search_preview keeps the field, because the gateway accepts it there", () => {
    const body = build("muse-spark-1.2-contributor", {
      tools: [{ ...webSearchTool(), type: "web_search_preview" }],
    });
    const tool = toolsOf(body)[0]!;
    expect(tool.type).toBe("web_search_preview");
    expect(tool.search_content_types).toEqual(["text", "image"]);
  });

  test("another model on the same provider is untouched", () => {
    const body = build("gpt-5.6-luna", { tools: [webSearchTool()] });
    expect(toolsOf(body)[0]!.search_content_types).toEqual(["text", "image"]);
  });

  test("a nested additional_tools declaration is sanitized too", () => {
    const body = build("muse-spark-1.2-contributor", {
      input: [{ type: "additional_tools", tools: [webSearchTool()] }],
    });
    const item = (body.input as Array<Record<string, unknown>>)[0]!;
    const nested = (item.tools as Array<Record<string, unknown>>)[0]!;
    expect(nested.type).toBe("web_search");
    expect(Object.hasOwn(nested, "search_content_types")).toBe(false);
  });

  test("the registry routes only the named exact models to Responses", () => {
    const defaults = getProviderRegistryEntry("opencode-go")?.modelWireDefaults ?? {};
    expect(defaults["muse-spark-1.2-contributor"]).toBe("openai-responses");
    // An exact-model allowlist, not a family rule: a sibling must not be dragged along.
    expect(defaults["muse-spark-1.2"]).toBeUndefined();
    // 1.3 serves the same Responses-only shape on Zen Go (probed: /chat/completions -> 500).
    expect(defaults["muse-spark-1.3-contributor"]).toBe("openai-responses");
  });

  test("1.3 gets the same web_search strip (probed: 1.3 + search_content_types -> 400)", () => {
    const body = build("muse-spark-1.3-contributor", { tools: [webSearchTool()] });
    const tool = toolsOf(body)[0]!;
    expect(tool.type).toBe("web_search");
    expect(Object.hasOwn(tool, "search_content_types")).toBe(false);
  });
});

/**
 * Zen Go rejects function names longer than 64 chars and recursive JSON schemas
 * (probed live: `name must be at most 64 characters, got 66` from Codex MCP tools
 * such as `muse-spark-web-search-compat`, and `Recursive JSON schemas are not
 * currently supported` from cyclic `$defs`). Dropping only the offending
 * declarations lets the turn proceed with the remaining catalog instead of
 * failing the whole request with a 400.
 */
describe("Muse Spark tool-surface compatibility", () => {
  const functionTool = (name: string, parameters: Record<string, unknown> = { type: "object" }) => ({
    type: "function",
    name,
    parameters,
  });

  test("drops function tools with names longer than 64 chars, keeps a 64-char name", () => {
    const body = build("muse-spark-1.3-contributor", {
      tools: [functionTool("a".repeat(65)), functionTool("b".repeat(64))],
    });
    const names = toolsOf(body).map(tool => tool.name);
    expect(names).toEqual(["b".repeat(64)]);
  });

  test("a nested additional_tools declaration is filtered too", () => {
    const body = build("muse-spark-1.3-contributor", {
      input: [{ type: "additional_tools", tools: [functionTool("c".repeat(66))] }],
    });
    const item = (body.input as Array<Record<string, unknown>>)[0]!;
    expect((item.tools as unknown[])).toEqual([]);
  });

  test("tool_choice naming a dropped tool falls back to auto", () => {
    const longName = "d".repeat(65);
    const body = build("muse-spark-1.3-contributor", {
      tools: [functionTool(longName)],
      tool_choice: { type: "function", name: longName },
    });
    expect(body.tool_choice).toBe("auto");
  });

  test("another model on the same provider keeps over-long names untouched", () => {
    const body = build("gpt-5.6-luna", { tools: [functionTool("e".repeat(65))] });
    expect(toolsOf(body).map(tool => tool.name)).toEqual(["e".repeat(65)]);
  });

  const cyclicParameters = () => ({
    type: "object",
    properties: { q: { $ref: "#/$defs/q" } },
    $defs: { q: { type: "object", properties: { sub: { $ref: "#/$defs/q" } } } },
  });

  test("drops function tools with cyclic local $refs", () => {
    const body = build("muse-spark-1.3-contributor", {
      tools: [functionTool("cyclic_tool", cyclicParameters()), functionTool("fine_tool")],
    });
    expect(toolsOf(body).map(tool => tool.name)).toEqual(["fine_tool"]);
  });

  test("keeps diamond $refs that share one $defs entry without cycling", () => {
    const diamond = {
      type: "object",
      properties: { a: { $ref: "#/$defs/x" }, b: { $ref: "#/$defs/x" } },
      $defs: { x: { type: "string" } },
    };
    const body = build("muse-spark-1.3-contributor", {
      tools: [functionTool("diamond_tool", diamond)],
    });
    expect(toolsOf(body).map(tool => tool.name)).toEqual(["diamond_tool"]);
  });

  test("catches recursion through $ref siblings (JSON Schema $ref-with-siblings form)", () => {
    const sneaky = {
      $ref: "#/$defs/base",
      properties: { loop: { $ref: "#" } },
      $defs: { base: { type: "object" } },
    };
    const body = build("muse-spark-1.3-contributor", {
      tools: [functionTool("sneaky_tool", sneaky), functionTool("fine_tool")],
    });
    expect(toolsOf(body).map(tool => tool.name)).toEqual(["fine_tool"]);
  });

  test("fails closed on over-budget $defs graphs instead of expanding exponentially", () => {
    // Each level references its predecessor twice: depth N costs ~2^N visits
    // without a bound. Depth 14 (~16k unbudgeted visits) must already drop.
    const defs: Record<string, unknown> = { d0: { type: "string" } };
    for (let i = 1; i <= 14; i += 1) {
      defs[`d${i}`] = {
        type: "object",
        properties: { left: { $ref: `#/$defs/d${i - 1}` }, right: { $ref: `#/$defs/d${i - 1}` } },
      };
    }
    const nested = { type: "object", properties: { root: { $ref: "#/$defs/d14" } }, $defs: defs };
    const body = build("muse-spark-1.3-contributor", {
      tools: [functionTool("nested_tool", nested), functionTool("fine_tool")],
    });
    expect(toolsOf(body).map(tool => tool.name)).toEqual(["fine_tool"]);
  });

  test("keeps a wide flat diamond that stays far under the walk budget", () => {
    const properties: Record<string, unknown> = {};
    for (let i = 0; i < 100; i += 1) properties[`p${i}`] = { $ref: "#/$defs/x" };
    const wide = { type: "object", properties, $defs: { x: { type: "string" } } };
    const body = build("muse-spark-1.3-contributor", {
      tools: [functionTool("wide_tool", wide)],
    });
    expect(toolsOf(body).map(tool => tool.name)).toEqual(["wide_tool"]);
  });

  test("another model on the same provider keeps cyclic schemas untouched", () => {
    const body = build("gpt-5.6-luna", {
      tools: [functionTool("cyclic_tool", cyclicParameters())],
    });
    expect(toolsOf(body).map(tool => tool.name)).toEqual(["cyclic_tool"]);
  });
});
