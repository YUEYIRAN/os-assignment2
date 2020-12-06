#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<limits.h>
#include<fcntl.h>
#include<unistd.h> 
#include<sys/mman.h>
#include<sys/stat.h>
#include<sys/types.h>

#define TLB_SIZE 16			//TLB条目数 
#define PAGES 256			//页条目数 
#define PAGE_SIZE 256		//页面字节大小
#define BUFFER_SIZE 10		//input文件中每行读入的最大的字符数量 
#define MEMORY_SIZE 65536
#define FALSE -1
#define FIFO 1
#define LRU 0

int TLB_Flag = FIFO;
int PageReplacement_Flag = FIFO;
int FRAMES = 256;
//TLB和Page Replacement的默认策略均为FIFO

//TLB结构
struct TLB {
    int Page;
    int Frame;
};//TLB结构&数组-FIFO

struct TLB_Node{
    struct TLB_Node* next; 
    struct TLB_Node* prior;
    int Page;
    int Frame;
};//双向链表-LRU

struct TLB_Linklist{
    struct TLB_Node* head;
    struct TLB_Node* rear;
};//LRU链表

struct TLB TLB_Array[TLB_SIZE];
struct TLB_Linklist* TLB_List;
int TLB_Index = 0;
//使用数组实现FIFO，使用链表实现LRU

//页表相关数据结构
struct Page_Table
{
    int Page;
    int Frame;
};//页表内容，同时也是数组方式的页表节点

struct Page_Table_Node
{
    struct Page_Table_Node* next;
    struct Page_Table_Node* prior;
    int Page;
    int Frame;
};//链表方式的页表节点，用于实现LRU算法

struct Page_Table_Linklist{
    struct Page_Table_Node* head;
    struct Page_Table_Node* rear;
};

struct Page_Table Page_Table_Array[PAGES];
struct Page_Table_Linklist* Page_Table_List;
int Page_Table_Index = 0;

//FIF0-TLB
int FIFO_IF_IN_TLB(int page){
    for(int i = 0; i < TLB_SIZE; i++){
        struct TLB* temp = &TLB_Array[i];
        if(temp->Page == page){//FIFO策略判断是否在当前TLB中
            return temp->Frame;
        }
    }
    return FALSE;
}

void Add_FIFO_To_TLB(int page, int frame){
    struct TLB* New_TLB = &TLB_Array[TLB_Index];
    TLB_Index = (TLB_Index + 1) % TLB_SIZE;
    New_TLB->Frame = frame;
    New_TLB->Page = page;
    //FIFO策略加入当前TLB
}
//FIFO-页表
int FIFO_IF_In_Page_Table(int page){//判断是否在当前页表中
    for(int i = 0; i < FRAMES; i++){
        struct Page_Table* temp = &Page_Table_Array[i];
        if(temp->Page == page){
            return temp->Frame;
        }
    }
    return FALSE;
}

int Add_FIFO_To_Page_Table(int page){//FIFO策略加入至当前Page Table中
    struct Page_Table* New_Page_Table = &Page_Table_Array[Page_Table_Index];

    New_Page_Table->Frame = Page_Table_Index;
    Page_Table_Index = (Page_Table_Index + 1) % FRAMES;
    New_Page_Table->Page = page;
    return New_Page_Table->Frame;
}

//LRU-TLB
int IN_TLB_LRU(int page){
    struct TLB_Node* temp = (struct TLB_Node*)malloc(sizeof(struct TLB_Node));
    temp = TLB_List->head->next;
    while(temp->next != NULL){
        if (temp->Page == page){//LRU策略判断是否在当前TLB中
            temp->prior->next = temp->next;
            temp->next->prior = temp->prior;
            //还原链表
            temp->prior = TLB_List->rear->prior;
            temp->next = TLB_List->rear;
            temp->prior->next = temp;
            TLB_List->rear->prior = temp;
            //temp指向的节点最近被调用，设置到最后
            return temp->Frame;
        }
        temp = temp->next;
    }
    return FALSE;
}

void Add_To_TLB_LRU(int page, int frame){//LRU策略加入当前TLB
    struct TLB_Node* temp = (struct TLB_Node*)malloc(sizeof(struct TLB_Node));
    temp->Page = page;
    temp->Frame = frame;
    if(TLB_Index == 0){//若链表为空，正常插入头
        TLB_Index++;

        temp->prior = TLB_List->head;
        temp->next = TLB_List->rear;
        TLB_List->head->next = temp;
        TLB_List->rear->prior = temp;
    }
    else if(TLB_Index < TLB_SIZE){
        //如果此时TLB链表非空未满，则正常插入到链表尾
        TLB_Index++;
        
        temp->prior = TLB_List->rear->prior;
        temp->next = TLB_List->rear;
        temp->prior->next = temp;
        TLB_List->rear->prior = temp;
    }
    else{
        //此时链表已满，需要先在尾端插入下一个节点
        temp->prior = TLB_List->rear->prior;
        temp->next = TLB_List->rear;
        temp->prior->next = temp;
        TLB_List->rear->prior = temp;
        //释放最近未调用的节点head->next
        TLB_List->head->next = TLB_List->head->next->next;
        free(TLB_List->head->next->prior);
        TLB_List->head->next->prior = TLB_List->head;
    }
}

//LRU-页表
int In_Page_Table_LRU(int page){
    struct Page_Table_Node* temp = (struct Page_Table_Node*)malloc(sizeof(struct Page_Table_Node));
    temp = Page_Table_List->head->next;
    while(temp->next != NULL){
        if (temp->Page == page){//判断是否在当前页表中
            temp->prior->next = temp->next;
            temp->next->prior = temp->prior;
            //还原原链表
            temp->prior = Page_Table_List->rear->prior;
            temp->next = Page_Table_List->rear;
            temp->prior->next = temp;
            Page_Table_List->rear->prior = temp;
            //temp指向的这个节点最近被调用，所以将其设置到最后，实现LRU的规则
            return temp->Frame;
        }
        temp = temp->next;
    }
    return FALSE;
}

int Add_To_Page_Table_LRU(int page){//加入当前Page Table中
    struct Page_Table_Node* temp = (struct Page_Table_Node*)malloc(sizeof(struct Page_Table_Node));
    temp->Page = page;
    if(Page_Table_Index == 0){
        //链表为空，在链表尾端正常插入下一个节点
        temp->Frame = Page_Table_Index;
        Page_Table_Index++;

        temp->prior = Page_Table_List->head;
        temp->next = Page_Table_List->rear;
        Page_Table_List->head->next = temp;
        Page_Table_List->rear->prior = temp;
    }
    else if(Page_Table_Index < FRAMES){
        //如果此时TLB链表为非空且未满，则正常插入到链表尾
        temp->Frame = Page_Table_Index;
        Page_Table_Index++;
        temp->prior = Page_Table_List->rear->prior;
        temp->next = Page_Table_List->rear;
        temp->prior->next = temp;
        Page_Table_List->rear->prior = temp;
    }
    else{//链表已满，需要先链表尾端插入下一个节点
        temp->Frame = Page_Table_List->head->next->Frame;
        temp->prior = Page_Table_List->rear->prior;
        temp->next = Page_Table_List->rear;
        temp->prior->next = temp;
        Page_Table_List->rear->prior = temp;
        //链表已满所以需要释放最近没有调用的
        //根据LRU的规则，此时距离上次被调用最久的节点位于head的next，故释放这个节点
        Page_Table_List->head->next = Page_Table_List->head->next->next;
        free(Page_Table_List->head->next->prior);
        Page_Table_List->head->next->prior = Page_Table_List->head;
    }
    return temp->Frame;
}


int main(int argc, char* argv[]){
    int opt;
	while ((opt = getopt(argc, argv, "t:p:n:")) != -1){
		switch (opt) {//默认页表缓存和页面切换策略为FIFO
			case 't':
           		if(strcmp(optarg, "LRU") == 0){
                       TLB_Flag = LRU;//切换TLB策略至LRU
                   }				
            	break;         
	        case 'p':
           		if(strcmp(optarg, "LRU") == 0){
                       PageReplacement_Flag = LRU;//切换页表切换策略至LRU
                   }					
            	break;           
            case 'n': 
            	if(strcmp(optarg, "256") == 0){
                    FRAMES = 256;//切换物理帧从128到256
                }
            	break;
	    }
	}

    const char* File1 = argv[1];//支持文档
    const char* File2 = argv[2];//地址文档
    int direction = open(File1, O_RDONLY);
	FILE* Input = fopen(File2, "r");
    signed char* Backing = mmap(0, MEMORY_SIZE, PROT_READ, MAP_PRIVATE, direction, 0);

    char Buffer[BUFFER_SIZE];//缓存
    int TLB_Hits = 0;//命中次数
	int Page_Faults = 0;//不在当前页中的次数
    signed char Main_Mem[FRAMES * PAGE_SIZE];//物理内存

    //页替换中FIFO下初始化
    if(PageReplacement_Flag == FIFO){
        int i = 0;
        for (i = 0; i < FRAMES; i++){//令页表各条目为-1
            Page_Table_Array[i].Frame = -1;
            Page_Table_Array[i].Page = -1;
        }
    }
    //TLB中LRU下初始化
    if(TLB_Flag == LRU)
	{
		TLB_List=(struct TLB_Linklist *)malloc(sizeof(struct TLB_Linklist));  
		TLB_List->head=(struct TLB_Node *)malloc(sizeof(struct TLB_Node));
		TLB_List->rear=(struct TLB_Node *)malloc(sizeof(struct TLB_Node));
		
		TLB_List->head->next = TLB_List->rear;
		TLB_List->head->prior = NULL;
		TLB_List->rear->next = NULL;
		TLB_List->rear->prior = TLB_List->head;	
        //尾指针指向头指针，表示此时链表为空
	}
    
    //Page Repalcement中LRU下初始化
    if(PageReplacement_Flag == LRU){
		Page_Table_List=(struct Page_Table_Linklist *)malloc(sizeof(struct Page_Table_Linklist));  
		Page_Table_List->head=(struct Page_Table_Node *)malloc(sizeof(struct Page_Table_Node));
		Page_Table_List->rear=(struct Page_Table_Node *)malloc(sizeof(struct Page_Table_Node));
		
		Page_Table_List->head->next = Page_Table_List->rear;
		Page_Table_List->head->prior = NULL;
		Page_Table_List->rear->next = NULL;
		Page_Table_List->rear->prior = Page_Table_List->head;
        //尾指针指向头指针，表示此时链表为空	
	}

    //读入addresses.txt
    while(fgets(Buffer, BUFFER_SIZE, Input) != NULL){
        int Virtual_Add = atoi(Buffer);
        int Physical_Add = 0;
        int offset = Virtual_Add % 256;//虚拟地址最右端的8位
        int page = (Virtual_Add >> 8) % 256;//虚拟地址次右端8位
        int frame;

        if(TLB_Flag = FIFO){
            frame = FIFO_IF_IN_TLB(page);//TLB
        }
        else{
            frame = IN_TLB_LRU(page);
        }
        if(frame != FALSE){//命中
            TLB_Hits++;
            Physical_Add = offset + 256 * frame;
            printf("Virtual_Address: %d, Physical_Address: %d, Value: %d\n", 
                    Virtual_Add, Physical_Add, Main_Mem[frame * PAGE_SIZE + offset]);
        }
        else{
            if(PageReplacement_Flag == FIFO){
                frame = FIFO_IF_In_Page_Table(page);
            }
            else{
                frame = In_Page_Table_LRU(page);
            }
            if (frame == FALSE){
                Page_Faults++;//Page Fault，向页表中添加节点
                if(PageReplacement_Flag == FIFO){
                    frame = Add_FIFO_To_Page_Table(page);
                }
                else{
                    frame = Add_To_Page_Table_LRU(page);
                }
                memcpy(Main_Mem + frame * PAGE_SIZE, Backing + page * PAGE_SIZE, PAGE_SIZE);
                //需要复制的文件来自backing file
            }Physical_Add = offset + 256 * frame;
            printf("Virtual_Address: %d, Physical_Address: %d, Value: %d\n", 
                    Virtual_Add, Physical_Add, Main_Mem[frame * PAGE_SIZE + offset]);

            if(TLB_Flag == FIFO)  Add_FIFO_To_TLB(page, frame);//FIFO
            else Add_To_TLB_LRU(page, frame);//LRU
        }
    }
    printf("Page Faults Rate= %d/1000\n", Page_Faults);
	printf("TLB Hits Rate= %d\1000n", TLB_Hits);
    return 0;
}

