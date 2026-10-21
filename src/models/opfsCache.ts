const ROOT_DIRECTORY = 'glitch-llm'

/** Same shape as tvmjs' ArtifactCache, so tvmjs can load weights straight out of OPFS. */
export interface OpfsCache {
  has(url: string): Promise<boolean>
  fetchWithCache(url: string): Promise<Response>
}

export const createOpfsCache = (scope: string): OpfsCache => {
  let scopeDirectory: Promise<FileSystemDirectoryHandle> | undefined

  const openDirectory = async () => {
    const root = await navigator.storage.getDirectory()
    const app = await root.getDirectoryHandle(ROOT_DIRECTORY, { create: true })
    return app.getDirectoryHandle(scope, { create: true })
  }

  const directory = () => (scopeDirectory ??= openDirectory())
  const fileName = (url: string) => encodeURIComponent(url)

  // An empty file is a download that died before its writable stream committed.
  const storedFile = async (url: string) => {
    try {
      const file = await (await (await directory()).getFileHandle(fileName(url))).getFile()
      return file.size > 0 ? file : undefined
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotFoundError') return undefined
      throw error
    }
  }

  const download = async (url: string) => {
    const response = await fetch(url)
    if (!response.ok || !response.body) throw new Error(`Download failed with ${response.status}: ${url}`)
    const handle = await (await directory()).getFileHandle(fileName(url), { create: true })
    await response.body.pipeTo(await handle.createWritable())
    return handle.getFile()
  }

  return {
    has: async (url) => (await storedFile(url)) !== undefined,
    fetchWithCache: async (url) => new Response((await storedFile(url)) ?? (await download(url))),
  }
}
