import { z } from "zod";

/**
 * Report configuration (FR4.2). All fields have defaults and are overridable.
 */
export interface ReportConfig {
  test_command: string | "auto";
  result_file: string | "auto";
  output_format: "markdown" | "html" | "json";
  output_path: string;
  coverage: "auto" | "on" | "off";
  fail_threshold?: number;
  mode: "execute" | "parse";
}

/**
 * Zod schema encoding FR4.2 defaults verbatim.
 */
export const ConfigSchema = z.object({
  test_command: z.union([z.literal("auto"), z.string()]).default("auto"),
  result_file: z.union([z.literal("auto"), z.string()]).default("auto"),
  output_format: z.enum(["markdown", "html", "json"]).default("markdown"),
  output_path: z.string().default("reports/"),
  coverage: z.enum(["auto", "on", "off"]).default("auto"),
  fail_threshold: z.number().min(0).max(100).optional(),
  mode: z.enum(["execute", "parse"]).default("execute"),
});

export const parse = ConfigSchema.parse;

/**
 * Resolve a partial user input into a fully-defaulted ReportConfig.
 *
 * `mode` is derived: parse-mode when `result_file !== "auto"` AND
 * `test_command === "auto"`; otherwise execute-mode. This satisfies FR1.3
 * (two work modes) and US4 (parse existing result files without re-running).
 */
export function resolveConfig(input: Partial<ReportConfig>): ReportConfig {
  const merged = { ...input };
  const config = ConfigSchema.parse(merged) as ReportConfig;

  // Derive mode unless the user explicitly forced it.
  const userSetMode = "mode" in merged && merged.mode !== undefined;
  if (!userSetMode) {
    if (config.result_file !== "auto" && config.test_command === "auto") {
      config.mode = "parse";
    } else {
      config.mode = "execute";
    }
  }
  return config;
}
