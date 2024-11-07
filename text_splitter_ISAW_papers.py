import re

def preprocess_text(text):
    # Remove single capital letters followed by a period and space or newline
    return re.sub(r'\b[A-Z]\.\s*', '', text)

def split_text_with_min_length(text, max_length=1200, min_length=400):
    # Split at sentence-ending periods followed by space and capital letter
    chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z][a-z])', text)
    
    # Ensure each chunk is within the max length by further splitting if necessary
    result = []
    for chunk in chunks:
        # If the chunk is longer than max_length, split it further
        while len(chunk) > max_length:
            # Find the last period within max_length to split, or split at max_length
            split_point = chunk.rfind('. ', 0, max_length) + 1 or max_length
            result.append(chunk[:split_point].strip())
            chunk = chunk[split_point:].strip()
        
        # Append the final chunk if it's within the limit
        if chunk:
            result.append(chunk)
    
    # Combine chunks to meet the minimum length requirement
    combined_result = []
    current_chunk = ""
    
    for chunk in result:
        if len(current_chunk) < min_length:
            current_chunk += (" " + chunk).strip()
        else:
            combined_result.append(current_chunk)
            current_chunk = chunk
    
    # Append any remaining text in the last chunk
    if current_chunk:
        combined_result.append(current_chunk)
    
    return combined_result

# Main function to run the entire process
def process_text(text, max_length=1200, min_length=300):
    # Step 1: Preprocess the text to remove single-letter abbreviations
    preprocessed_text = preprocess_text(text)
    
    # Step 2: Split the text with min and max length constraints
    return split_text_with_min_length(preprocessed_text, max_length, min_length)

# Example usage
text = """
ISAW Papers 7.8 (2014)\n\nISAW Papers: Towards a Journal as Linked Open Data\n\nSebastian Heath\n\nIntroduction\n\nThe present contribution to the set of essays published under the rubric of \u201cISAW Papers 7\u201d is necessarily self-referential. ISAW Papers, the Institute for the Study of the Ancient World\u2019s digital scholarly journal, is both its topic and its venue. These overlapping roles will prove useful by allowing direct illustration of the progress ISAW has made in implementing the goals with which the journal was initiated. By way of high-level overview, those goals are to publish article-length scholarship that (1) is available at no-cost to readers, (2) that can be reused and redistributed under a Creative Commons license, and (3) that is stored in formats that are very likely to be readable into the far future. Additionally, articles in ISAW Papers should link to stable resources similarly available on the public Internet. This last goal is intended to increase the discoverability and utility of any individual article as well as of the growing network of digital resources available for investigating the ancient world.\u2b08#p1\n\nIn describing progress to date, the following paragraphs will not shy away from raising technical issues. They do not, however, offer complete instructions for deploying Linked Open Data in a journal context nor detailed introductions to the technologies described. The discussion is practice oriented and so makes reference to the articles published to date. This approach and the movement from overview to specifics is intended to introduce readers to some of the opportunities ISAW Papers has recognized and also to the challenges it faces.\u2b08#p2\n\nTo start broadly, the editorial scope of ISAW Papers is as wide as ISAW\u2019s intellectual mission, which itself embraces \u201cthe development of cultures and civilizations around the Mediterranean basin, and across central Asia to the Pacific Ocean.\u201d (ISAW n.d.) Temporally, ISAW is mainly concerned with complex cultures before the advent of early modern globalization. Though it is important to note that ISAW does not try to impose strict limits on what falls within its intellectual purview. Indeed, the origins, development and reception of all phases of the Ancient World are fair game at ISAW.\u2b08#p3\n\nReview and Licensing\n\nTwo additional concerns of a scholarly journal - review and licensing - can also be addressed efficiently. ISAW Papers publishes anonymously peer-reviewed articles as well as articles read and forwarded for publication by members of the ISAW faculty. This aspect of the editorial process is made clear for each article. The goal here is to provide a balance between the many benefits that peer review can provide to an author while similarly ensuring that it is neither a barrier to new work nor an impediment to timely publication. In terms of licensing, ISAW asks authors to agree to distribution of their text under a Creative Commons Attribution (CC-BY) license. The same applies to images authors have created on their own or which ISAW creates during the editorial process. We consider such open distribution to be an important component of a robust approach to future accessibility. It is, however, the case that authors have needed to include images whose copyright is held by others. This situation remains a fact of public scholarly discourse. Accordingly, we ask that authors obtain permission for ISAW to publish such images in digital form but do not require explicit agreement to a CC license. As with peer-review, a reasonable balance of current realities and future possibilities is the goal.\u2b08#p4\n\nPartnership with the NYU Library\n\nInitial public availability takes place in partnership with the New York University (NYU) Library. So for example, the text you are reading now will be accessible via the URI \u201chttp://dlib.nyu.edu/awdl/isaw/isaw-papers/7/heath/\u201d. While ISAW has complete responsibility for the editorial process, that is for shepherding an author\u2019s intellectual content into a form that enables both long-term accessibility and immediate distribution, we rely on the Library to provide the infrastructure for that long-term preservation. Each party in this relationship brings its institutional strengths to the endeavor. In particular, it is very useful that the library assigns a Handle to each article (CNRI n.d.). For example, the URL \u201chttp://hdl.handle.net/2333.1/k98sf96r\u201d will redirect to whichever URL the NYU Library is using to host ISAW Papers\u2019 first article (Jones and Steele 2011). If a reader follows that link within a few years of the publication of this current discussion, it is likely she or he will be redirected to \u201chttp://dlib.nyu.edu/awdl/isaw/isaw-papers/1/.\u201d Further out into the future, the handle may resolve to a different address. But we at ISAW are confident that an institution such as the NYU Library offers a very strong likelihood of ongoing availability. And it is of course the case that we encourage readers and other institutions to download and re-distribute any and all ISAW Papers articles. Such third-party use and archiving, enabled through initial distribution by the Library, will also contribute to the long-term preservation of this content.\u2b08#p5\n\nAn additional result of collaboration with NYU Library staff, particularly my colleagues in the ISAW library, is the creation of individual records in the NYU Bobcat library catalog for each article. This local initiative leads in turn and automatically to the creation of a Worldcat record for each article. Accordingly, \u201chttp://www.worldcat.org/oclc/811756919\u201d is the Worldcat \u201cpermalink\u201d for the record describing C. Lorber and A. Meadow\u2019s 2012 review of Ptolemaic numismatics. The journal itself has a Library of Congress issued International Standard Serial Number (2164-1471) as well as its own Worldcat record at \u201chttp://www.worldcat.org/oclc/756047783\u201d.\u2b08#p6\n\nBroad Strokes and Specific Citations\n\nThere is a future point at which the following short list will describe the main components of a born-digital article published in ISAW Papers:\u2b08#p7\n\nAn archival version in well-crafted XHTML5 that is available through the NYU Faculty Digital Archive (http://archive.nyu.edu).\n\nLinks to stable external resources encoded using RDFa, a widely supported standard that is discussed below.\n\nThe NYU Library will provide access to a version of the document formatted for reading and with additional User Interface (UI) elements that encourage engagement with the content.\n\nThe two new abbreviations in the above list - XHTML and RDFa - can bear further explanation. As is probably well-known to many readers, HTML, specifically its 5th version HTML5, is the standard published by the Worldwide Web Consortium (W3C) that specifies the format of text intended for transmission from Internet servers to web browsers. As a simple description, HTML allows content-creators to specify the visible aspects of a text: e.g., that titles and headings are in bold, that paragraphs are visually distinct by indentation or spacing, and other aspects such as italic or bold spans. For its part, the W3C has quickly become a standards-setting body with global impact. At this moment, HTML5 documents can be directly read - that is rendered into human readable form on screen - by many applications running on many different forms of computing devices ranging from desktops and notebook computers to tablets and phones. It is likely that this easy readability of HTML documents will continue far into the future and ISAW believes some degree of readability for such content is guaranteed in perpetuity to the extent that that can be reasonably foreseen.\u2b08#p8\n\nXHTML is the variant of HTML that adheres strictly to the requirements of the Extensible Markup Language (XML). XML is in turn a standard that provides more explicit indications of the structure of a text than does HTML. For example, an item in a list in HTML can be indicated by \u201c<li>An item in a list\u201d, whereas XHTML requires that the markup be \u201c<li>An item in a list</li>\u201d. Note the terminating \u201c</li>\u201d, which is required in XML. While a full discussion of XML and XHTML would take up excessive room here, it is fair to say that their added requirements are geared towards enabling more reliable processing by automated agents, meaning the manipulation of the text and rendering of results by computer programs.\u2b08#p9\n\nAt this point in the discussion it is worth highlighting one particular aspect of XHTML that ISAW Papers utilizes extensively. On the public internet, the presence of a \u201cpound sign\u201d or \u201c#\u201d in a web address often indicates a reference to a particular part of a document. When used in this way, the exact part referenced is indicated in the HTML document itself by the presence of an \u2018id\u2019 attribute. Meaning that HTML\u2019s \u2018p\u2019 element, which is used to mark paragraphs, can be identified by mark up of the form \u2018<p id=\u201dp10\u201d> \u2026 </p>\u2019. In ISAW Papers, all paragraphs in the main body of an article have such an id and can therefore be directly referenced via URLs. For example \u201chttp://dlib.nyu.edu/awdl/isaw/isaw-papers/6/#p3\u201d is a direct link to the third paragraph of M. Zarmakoupi\u2019s (2013) article on urban development in Hellenistic Delos.\u2b08#p10\n\nTowards Linked Open Data\n\nMost of the discussion so far should be considered as preliminary to a focus on ISAW Paper\u2019s implementation of the principles of Linked Open Data (LOD), principles that were summarized in the Introduction to this set of articles. With that description in mind, ISAW Papers can make some claim to being \u201c5 Star\u201d linked data as defined in Berners-Lee\u2019s fundamental note of 2006. 

"""
# Run the process
output_chunks = process_text(text)

for c in output_chunks:
    print(c)
    print("------------------------------------------")   