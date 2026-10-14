import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, describeError } from "./api";

afterEach(() => vi.unstubAllGlobals());

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue(new Response(body === undefined ? null : JSON.stringify(body), { status }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("describeError", () => {
  it("formats FastAPI validation errors", () => {
    expect(describeError([{ loc: ["body", "name"], msg: "Field required" }])).toBe("name: Field required");
  });
  it("formats duplicate-image conflicts", () => {
    const msg = describeError({ message: "Duplicate.", conflicts: [{ filename: "a.jpg", object_id: "ODNA-1", object_name: "Mug" }] });
    expect(msg).toContain("a.jpg is already a reference of Mug (ODNA-1)");
  });
  it("formats rejected files", () => {
    expect(describeError({ message: "Rejected.", files: [{ filename: "x.txt", error: "bad" }] })).toContain("x.txt: bad");
  });
});

describe("api client", () => {
  it("returns parsed JSON on success", async () => {
    const f = mockFetch(200, { status: "ok", registered_objects: 3 });
    const h = await api.health();
    expect(h.registered_objects).toBe(3);
    expect(f).toHaveBeenCalledWith("/api/health", undefined);
  });

  it("throws ApiError with the server detail", async () => {
    mockFetch(409, { detail: { message: "Already registered." } });
    await expect(api.getObject("ODNA-X")).rejects.toMatchObject({ status: 409, message: "Already registered." });
  });

  it("reports an unreachable backend clearly", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const err = await api.dashboard().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toMatch(/Cannot reach/);
  });

  it("sends identification uploads as multipart form data", async () => {
    const f = mockFetch(200, { status: "NO_RELIABLE_MATCH", candidates: [] });
    await api.identify(new File([new Uint8Array([1, 2, 3])], "q.jpg", { type: "image/jpeg" }), 3);
    const [path, init] = f.mock.calls[0];
    expect(path).toBe("/api/identify");
    expect(init.method).toBe("POST");
    const fd = init.body as FormData;
    expect((fd.get("file") as File).name).toBe("q.jpg");
    expect(fd.get("top_k")).toBe("3");
  });

  it("handles 204 responses for deletion", async () => {
    mockFetch(204, undefined);
    await expect(api.deleteObject("ODNA-1")).resolves.toBeUndefined();
  });

  it("builds query strings without empty params", async () => {
    const f = mockFetch(200, { items: [], total: 0, page: 1, page_size: 24, categories: [] });
    await api.listObjects({ q: "mug", category: "", page: 2 });
    expect(f.mock.calls[0][0]).toBe("/api/objects?q=mug&page=2");
  });
});
