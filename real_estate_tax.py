"""
PDF 문서를 처리하여 벡터 저장소에 저장하는 모듈

이 모듈은 다음 단계를 수행합니다:
1. PDF를 OCR을 사용하여 마크다운으로 변환
2. 마크다운을 텍스트로 변환
3. 텍스트를 청크로 분할
4. 임베딩을 생성하여 벡터 저장소에 저장
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List

import markdown
import nest_asyncio
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyzerox import zerox

# 비동기 루프 중첩 허용
nest_asyncio.apply()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFToVectorStoreProcessor:
    """PDF 문서를 벡터 저장소로 변환하는 프로세서"""
    
    def __init__(
        self,
        pdf_path: str,
        output_dir: str = "./documents",
        collection_name: str = "document_collection",
        persist_directory: str = "./vector_store",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large",
        chunk_size: int = 1500,
        chunk_overlap: int = 100,
    ):
        """
        Args:
            pdf_path: PDF 파일 경로
            output_dir: 마크다운 파일 저장 디렉토리
            collection_name: Chroma 컬렉션 이름
            persist_directory: 벡터 저장소 저장 디렉토리
            model: Zerox OCR에 사용할 비전 모델
            embedding_model: OpenAI 임베딩 모델
            chunk_size: 텍스트 청크 크기
            chunk_overlap: 텍스트 청크 오버랩 크기
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.model = model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 파일명에서 확장자 제거
        self.base_name = self.pdf_path.stem
        self.markdown_path = self.output_dir / f"{self.base_name}.md"
        self.text_path = self.output_dir / f"{self.base_name}.txt"
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
    
    async def pdf_to_markdown(
        self,
        select_pages: Optional[List[int]] = None,
        custom_system_prompt: Optional[str] = None
    ) -> dict:
        """
        PDF를 OCR을 사용하여 마크다운으로 변환
        
        Args:
            select_pages: 처리할 페이지 번호 리스트 (None이면 모든 페이지)
            custom_system_prompt: 커스텀 시스템 프롬프트
            
        Returns:
            Zerox 처리 결과
        """
        logger.info(f"PDF를 마크다운으로 변환 중: {self.pdf_path}")
        
        result = await zerox(
            file_path=str(self.pdf_path),
            model=self.model,
            output_dir=str(self.output_dir),
            custom_system_prompt=custom_system_prompt,
            select_pages=select_pages
        )
        
        logger.info(f"마크다운 파일 생성 완료: {self.markdown_path}")
        return result
    
    def markdown_to_text(self) -> None:
        """마크다운 파일을 일반 텍스트로 변환"""
        logger.info(f"마크다운을 텍스트로 변환 중: {self.markdown_path}")
        
        if not self.markdown_path.exists():
            raise FileNotFoundError(f"마크다운 파일을 찾을 수 없습니다: {self.markdown_path}")
        
        # 마크다운 파일 읽기
        with open(self.markdown_path, 'r', encoding='utf-8') as md_file:
            md_content = md_file.read()
        
        # 마크다운을 HTML로 변환 후 텍스트 추출
        html_content = markdown.markdown(md_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text()
        
        # 텍스트 파일로 저장
        with open(self.text_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text_content)
        
        logger.info(f"텍스트 파일 생성 완료: {self.text_path}")
    
    def create_vector_store(self) -> Chroma:
        """
        텍스트를 청크로 분할하고 벡터 저장소 생성
        
        Returns:
            생성된 Chroma 벡터 저장소
        """
        logger.info("벡터 저장소 생성 중...")
        
        if not self.text_path.exists():
            raise FileNotFoundError(f"텍스트 파일을 찾을 수 없습니다: {self.text_path}")
        
        # 텍스트 스플리터 생성
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=['\n\n', '\n']
        )
        
        # 텍스트 로드 및 분할
        loader = TextLoader(str(self.text_path))
        document_list = loader.load_and_split(text_splitter)
        
        logger.info(f"문서를 {len(document_list)}개의 청크로 분할했습니다")
        
        # 임베딩 생성
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        
        # 벡터 저장소 생성
        vector_store = Chroma.from_documents(
            documents=document_list,
            embedding=embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory)
        )
        
        logger.info(f"벡터 저장소 생성 완료: {self.persist_directory}")
        return vector_store
    
    async def process(
        self,
        select_pages: Optional[List[int]] = None,
        custom_system_prompt: Optional[str] = None
    ) -> Chroma:
        """
        전체 처리 파이프라인 실행
        
        Args:
            select_pages: 처리할 페이지 번호 리스트
            custom_system_prompt: 커스텀 시스템 프롬프트
            
        Returns:
            생성된 벡터 저장소
        """
        try:
            # 1. PDF를 마크다운으로 변환
            await self.pdf_to_markdown(select_pages, custom_system_prompt)
            
            # 2. 마크다운을 텍스트로 변환
            self.markdown_to_text()
            
            # 3. 벡터 저장소 생성
            vector_store = self.create_vector_store()
            
            logger.info("모든 처리가 완료되었습니다!")
            return vector_store
            
        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            raise


async def main():
    """메인 실행 함수"""
    # 환경 변수 로드
    load_dotenv()
    
    # 프로세서 생성 및 실행
    processor = PDFToVectorStoreProcessor(
        pdf_path="./income_tax.pdf",
        output_dir="./documents",
        collection_name="income_tax_collection",
        persist_directory="./income_tax_collection",
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-large",
        chunk_size=1500,
        chunk_overlap=100
    )
    
    vector_store = await processor.process()
    return vector_store


if __name__ == "__main__":
    # 프로그램 실행
    result = asyncio.run(main())