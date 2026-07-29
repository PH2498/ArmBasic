import { describe, it, expect } from "vitest";
import { resolveConfig } from "../src/config.js";

describe("resolveConfig", () => {
  it("returns all defaults when no input", () => {
    const c = resolveConfig({});
    expect(c.output_format).toBe("markdown");
    expect(c.output_path).toBe("reports/");
    expect(c.coverage).toBe("auto");
    expect(c.test_command).toBe("auto");
    expect(c.mode).toBe("execute");
  });

  it("infers parse mode when result_file explicit and test_command auto", () => {
    const c = resolveConfig({ result_file: "reports/junit.xml" });
    expect(c.mode).toBe("parse");
  });

  it("honors explicit override", () => {
    const c = resolveConfig({ output_format: "html", fail_threshold: 80 });
    expect(c.output_format).toBe("html");
    expect(c.fail_threshold).toBe(80);
  });
});
