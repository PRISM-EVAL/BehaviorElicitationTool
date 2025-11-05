import asyncio
from typing import Awaitable, Callable, List, Tuple

import minillmlib as mll
from pymongo.collection import Collection

from bet.utils import logger

from .Factory import Factory

# Type aliases for transform callables
IndividualTransformCallable = Callable[
    [List[str], List[str], Collection, dict | None, mll.GeneratorInfo | None, int],
    Awaitable[List[str]]
]

GlobalTransformCallable = Callable[
    [str, Tuple[List[str], List[str]], Collection, dict | None, mll.GeneratorInfo | None, int],
    Awaitable[str]
]


class Transform:
    def __init__(self, 
        parent: Factory
    ):
        self.parent = parent

    def __str__(self) -> str:
        return str(self.parent)

    async def transform_with_retry(self,
        content: str,
        previous_transform: Tuple[List[str], List[str]],
        prompt_item_collection: Collection,
        data_buffer: dict | None = None,
        assistant: mll.GeneratorInfo | None = None,
        retry: int = 2,
        index: int = 0,
    ) -> str:
        for _ in range(retry):
            try:
                return await self.transform(
                    content=content,
                    previous_transform=previous_transform,
                    prompt_item_collection=prompt_item_collection,
                    data_buffer=data_buffer,
                    assistant=assistant,
                    index=index
                )
            except Exception as e:
                logger.debug(
                    {
                        "type": "transform_retry_error",
                        "factory": self.parent.full_name(),
                        "error": str(e),
                    }
                )
        logger.debug(
            {
                "type": "transform_failure",
                "message": f"Failed to transform after {retry} retries. Returning untransformed content",
            }
        )
        return content

    async def apply_transform_lst(self,
        content: List[str],
        previous_transform: Tuple[List[str], List[str]],
        prompt_item_collection: Collection,
        data_buffer: dict | None = None,
        assistant: mll.GeneratorInfo | None = None,
    ) -> Awaitable[List[str]]:
        tasks = [
            self.transform_with_retry(
                content=_content, 
                previous_transform=previous_transform, 
                prompt_item_collection=prompt_item_collection, 
                data_buffer=data_buffer, 
                assistant=assistant,
                index=index
            )
            for index, _content in enumerate(content)
        ]
        try:
            result = await asyncio.gather(*tasks)
            return result
        except Exception as e:
            logger.error(
                {
                    "type": "transform_error",
                    "error": str(e),
                    "factory": self.parent.full_name(),
                }
            )
            raise e


class IndividualTransform(Transform):
    def __init__(self,
        parent: Factory,
        transform: IndividualTransformCallable,
    ) -> None:
        self.transform: IndividualTransformCallable = transform
        super().__init__(parent=parent)


class GlobalTransform(Transform):
    def __init__(self,
        parent: Factory,
        transform: GlobalTransformCallable,
        applied: bool = False,
    ) -> None:
        self.transform: GlobalTransformCallable = transform
        self.applied = applied
        super().__init__(parent=parent)

    async def apply_transform_lst(self,
        content: List[str],
        previous_transform: Tuple[List[str]],
        prompt_item_collection: Collection,
        data_buffer: dict | None = None,
        assistant: mll.GeneratorInfo | None = None
    ) -> Awaitable[List[str]]:
        self.applied = True
        return await super().apply_transform_lst(
            content=content, 
            previous_transform=previous_transform, 
            prompt_item_collection=prompt_item_collection,
            data_buffer=data_buffer,
            assistant=assistant,
        )
