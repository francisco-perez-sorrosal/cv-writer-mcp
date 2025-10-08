.PHONY: build-wheel build-mcpb clean help

help:
	@echo "CV Writer MCP Bundle Targets:"
	@echo "  build-wheel    - Build Python wheel distribution"
	@echo "  build-mcpb     - Create MCPB bundle with dependencies"
	@echo "  clean          - Remove build artifacts"

build-wheel:
	pixi run python-bundle

build-mcpb: build-wheel
	pixi run update-mcpb-deps
	pixi run mcp-bundle
	pixi run pack

clean:
	pixi run clean-bundles
